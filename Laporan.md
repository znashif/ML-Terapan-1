# Laporan Proyek Machine Learning - Zuhair Nashif Abdurrohim

## Domain Proyek

Kualitas buah merupakan salah satu faktor kunci dalam rantai pasok produk pertanian, terutama bagi produsen dan distributor besar seperti supermarket atau perusahaan ekspor. Penilaian kualitas buah secara manual rentan terhadap subjektivitas dan inkonsistensi. Oleh karena itu, pengembangan sistem prediksi kualitas buah berbasis data menjadi penting untuk memastikan standar mutu yang konsisten dan efisien.

Masalah ini perlu diselesaikan karena:
- Dapat membantu produsen mengotomatisasi proses sortasi buah.
- Mengurangi biaya dan waktu sortasi manual.
- Meningkatkan kepuasan konsumen dengan jaminan kualitas yang konsisten.

Penelitian terkait klasifikasi kualitas buah menunjukkan bahwa algoritma machine learning seperti **KNN dan Random Forest** efektif dalam mengklasifikasikan buah berdasarkan fitur seperti tingkat kemanisan, keasaman, kerenyahan, dan lainnya.

Referensi:  
[A Study of Combining KNN and ANN for Classifying Dragon Fruits Automatically](https://www.researchgate.net/profile/Nguyen-Trieu-11/publication/358940623_A_Study_of_Combining_KNN_and_ANN_for_Classifying_Dragon_Fruits_Automatically/links/648b02a17fcc811dcdd04bbb/A-Study-of-Combining-KNN-and-ANN-for-Classifying-Dragon-Fruits-Automatically.pdf)

## Business Understanding

### Problem Statements
1. Bagaimana cara mengklasifikasikan kualitas buah apel secara otomatis berdasarkan karakteristik fisik dan sensorik seperti sweetness, crunchiness, dan acidity?
2. Algoritma machine learning apa yang memberikan performa terbaik dalam mengklasifikasikan kualitas buah apel?
### Goals
1. Membangun model klasifikasi untuk memprediksi kualitas buah apel ( good / bad ) berdasarkan fitur numerik yang tersedia.
2. Membandingkan performa dua algoritma klasifikasi: K-Nearest Neighbor (KNN) dan Random Forest, menggunakan metrik akurasi dan F1-score.
### Solution Statements
- Menggunakan dua algoritma yaitu **K-Nearest Neighbor** dan **Random Forest** untuk membangun model klasifikasi kualitas buah.
- Melakukan evaluasi model menggunakan metrik seperti **accuracy**, **precision**, **recall**, dan **F1-score**.
- Menyempurnakan baseline model dengan **hyperparameter tuning** (misalnya memilih jumlah tetangga terbaik untuk KNN dan jumlah pohon pada Random Forest).
- Menyajikan perbandingan hasil model untuk merekomendasikan algoritma terbaik.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah dataset **Apple Quality** yang disediakan oleh Nidula Elgitiyewithana di Kaggle. Dataset ini berisi informasi mengenai karakteristik fisik dan sensorik dari berbagai buah apel untuk menentukan kualitasnya. Dataset ini telah melalui proses pembersihan dan skala ulang (scaled and cleaned), sehingga siap digunakan untuk keperluan analisis dan pemodelan.

Sumber dataset:  
[Apple Quality](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/data)

### Variabel-variabel pada Apple Quality dataset adalah sebagai berikut:
- **A_id** : Merupakan ID unik untuk setiap buah apel. (Tidak digunakan untuk pemodelan karena tidak bersifat prediktif)
- **Size** : Ukuran dari buah apel.
- **Weight** : Berat dari buah apel.
- **Sweetness** : Tingkat kemanisan buah apel.
- **Crunchiness** : Tingkat kerenyahan buah apel.
- **Juiciness** : Seberapa berair buah apel tersebut.
- **Ripeness** : Tahapan kematangan buah apel.
- **Acidity** : Tingkat keasaman buah apel.
- **Quality** : Label target/output yang merepresentasikan kualitas keseluruhan dari apel (misalnya _Low_, _Medium_, _High_ — akan dicek format pastinya). Fitur ini akan digunakan sebagai **label klasifikasi** dalam proyek ini.
  ![Pasted image 20250422223940](https://github.com/user-attachments/assets/a1a3ecf1-62d9-4db7-9917-9f9ba4c828c4)


### Ukuran Data
- Terdiri dari 4001 baris data dengan 9 kolom
- Terdapat 1 data NaN

  ![Pasted image 20250422224004](https://github.com/user-attachments/assets/695c32e8-fe36-433d-b8ae-4ea4ba99aaca)
- Hanya kolom Acidity dan Quality yang bertipe data object, sisanya bertipe float64

  ![Pasted image 20250422223940](https://github.com/user-attachments/assets/c2fc64d4-89c0-4b87-b5ff-d41572a3b706)
- Terdapat beberapa outlier

  ![Pasted image 20250422224041](https://github.com/user-attachments/assets/c5cc0f2e-0027-4511-be0d-13ca7f4def85)

- Persebaran Qualitas merata

  ![Pasted image 20250422224134](https://github.com/user-attachments/assets/2c317754-06aa-4f09-bec2-3ad384118cc2)

- Fitur lain juga terdistribusi normal

  ![Pasted image 20250422224205](https://github.com/user-attachments/assets/4e23da67-3538-4c9d-bf42-b7d44ce957ff)

- Berikut Heatmap untuk hubungan antar 2 fitur

  ![Pasted image 20250422224235](https://github.com/user-attachments/assets/f2b4edca-ddf4-4cd0-90d9-0f1fa689fd12)

- Dan scatter plot untuk multivariate analysis

  ![Pasted image 20250422224330](https://github.com/user-attachments/assets/b4ca94df-a6a4-4b80-81c2-3428a71e068b)


## Data Preparation
Berikut adalah teknik data preparation yang saya pakai
-  Melakukan penghapusan data kosong / NaN
	- Alasan : karena ML tidak bisa memproses data kosong / NaN dan juga bisa menyebabkan error atau hasil yang tidak akurat
- Atasi outlier dengan Winsorizing
	- Mengganti nilai outlier dengan batas atas atau bawah IQR
	- Alasan : untuk membersihkan data dan menghindari bias dan noise
- Melalukan Drop kolom / fitur ID
	- Alasan : karena mengandung informasi yang tidak relevan
- Encoding kolom Quality
	- Menggunakan label encoder untuk merubah nilai Quality menjadi 1 / 0
	- Alasan : karena ML membutuhkan data numerik (hanya bisa memproses data numerik)
- Split data
	- Melakukan split data dengan persentase 80:20 antara train dan test
	- Alasan : persiapan membuat model
- Standarisasi
	- Alasan : membuat data memiliki skala yang sama 

## Modeling
Tahapan ini membahas mengenai model _machine learning_ yang digunakan untuk menyelesaikan permasalahan, yaitu memprediksi kualitas apel ('Quality'). Dua model yang digunakan adalah K-Nearest Neighbors (KNN) dan Random Forest. Mari kita bahas tahapan dan parameternya:

### K-Nearest Neighbors (KNN)
Tahapan:
1. **Inisiasi Model:** Membuat objek model KNN dengan parameter `n_neighbors=10`. Parameter ini menentukan jumlah tetangga terdekat yang akan dipertimbangkan dalam proses prediksi.
2. **Training:** Model KNN dilatih menggunakan data latih (X_train dan y_train) dengan memanggil fungsi `fit()`.
Parameter:
- `n_neighbors=10`: Jumlah tetangga terdekat yang dipertimbangkan.
Kelebihan:
- Mudah diimplementasikan dan dipahami.
- Tidak memerlukan asumsi tentang distribusi data.
Kekurangan:
- Dapat terpengaruh oleh fitur yang tidak relevan.
- Komputasi dapat menjadi mahal untuk dataset yang besar.
### Random Forest
Tahapan:
1. **Inisiasi Model:** Membuat objek model Random Forest dengan parameter `n_estimators=50`, `max_depth=10`, `random_state=42`, dan `n_jobs=-1`.
2. **Training:** Model Random Forest dilatih menggunakan data latih (X_train dan y_train) dengan memanggil fungsi `fit()`.
Parameter:
- `n_estimators=50`: Jumlah pohon keputusan dalam _ensemble_.
- `max_depth=10`: Kedalaman maksimum setiap pohon keputusan.
- `random_state=42`: Untuk memastikan reproduktifitas hasil.
- `n_jobs=-1`: Menggunakan semua _core_ prosesor untuk mempercepat proses training.
Kelebihan:
- Mampu menangani data dengan banyak fitur dan _outlier_.
- Lebih robust dan akurat dibandingkan dengan pohon keputusan tunggal.
Kekurangan:
- Dapat menjadi _overfitting_ jika tidak diparameterisasi dengan baik.
- Lebih kompleks dan sulit diinterpretasi dibandingkan dengan KNN.

### Pemilihan Model Terbaik

Berdasarkan hasil evaluasi menggunakan metrik MSE, akurasi, presisi, _recall_, dan F1-Score, **model Random Forest menunjukkan performa yang lebih baik** dibandingkan dengan KNN. Hal ini terlihat dari nilai MSE yang lebih rendah dan skor evaluasi lainnya yang lebih tinggi pada data _testing_.

Alasan memilih Random Forest:
- Random Forest cenderung lebih robust dan mampu menangani kompleksitas data yang lebih tinggi dibandingkan dengan KNN.
- Random Forest menghasilkan prediksi yang lebih akurat pada data _testing_, yang mengindikasikan kemampuan generalisasi yang lebih baik.

## Evaluation

Bagian ini membahas metrik evaluasi yang digunakan untuk mengukur performa model _machine learning_ dalam memprediksi kualitas apel. Metrik yang digunakan adalah:
- **MSE (Mean Squared Error)**
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### Penjelasan Metrik
1. **MSE (Mean Squared Error):**
    - Digunakan untuk mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual.
    - Semakin kecil nilai MSE, semakin baik performa model.
    - Formula:
$ MSE = 1/n * Σ(y_i - ŷ_i)^2 $
di mana:    
- $n$ adalah jumlah data    
- $y_i$ adalah nilai aktual    
- $ŷ_i$ adalah nilai prediksi

2. **Accuracy:**
    - Mengukur proporsi prediksi yang benar dari keseluruhan data.
    - Formula:
$$
Accuracy = (TP + TN) / (TP + TN + FP + FN)
$$
di mana:    
- TP: *True Positive* (prediksi benar, kelas positif)    
- TN: *True Negative* (prediksi benar, kelas negatif)    
- FP: *False Positive* (prediksi salah, kelas positif)    
- FN: *False Negative* (prediksi salah, kelas negatif)

3. **Precision:**
    - Mengukur proporsi prediksi positif yang benar dari keseluruhan prediksi positif.
    - Formula:
$$
Precision = TP / (TP + FP)
$$

4. **Recall:**
    - Mengukur proporsi prediksi positif yang benar dari keseluruhan data yang sebenarnya positif.
    - Formula:
$$
Recall = TP / (TP + FN)
$$

5. **F1-Score:**
    - Merupakan rata-rata harmonik dari _precision_ dan _recall_.
    - Memberikan keseimbangan antara _precision_ dan _recall_.
    - Formula:
$$
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
$$

### Hasil Proyek Berdasarkan Metrik Evaluasi

|               | Accuracy | Precision | Recall   | F1-Score |
| ------------- | -------- | --------- | -------- | -------- |
| KNN           | 0.9      | 0.920844  | 0.874687 | 0.897172 |
| Random Forest | 0.89875  | 0.889706  | 0.909774 | 0.899628 |

- **KNN** memiliki keunggulan dalam **Accuracy** dan **Precision**, menunjukkan bahwa ia lebih baik dalam hal mengidentifikasi label positif dengan akurat.
- **Random Forest** unggul dalam **Recall** dan **F1-Score**, yang menunjukkan kemampuannya lebih baik dalam menangkap semua kasus positif dan memiliki keseimbangan antara precision dan recall.

Secara keseluruhan, jika menginginkan keseimbangan antara precision dan recall, **Random Forest** mungkin sedikit lebih baik karena F1-Score yang lebih tinggi.


## Business Understanding (Tambahan Hubungan dengan Bisnis/Kehidupan Nyata)

Proyek ini memiliki hubungan langsung dengan kebutuhan industri agrikultur dan rantai pasok distribusi buah, khususnya untuk produsen apel, distributor, supermarket, dan eksportir. Dengan adanya sistem klasifikasi kualitas apel berbasis machine learning:

- **Produsen** dapat mengotomatisasi proses sortasi kualitas, mengurangi ketergantungan pada tenaga kerja manual yang sering subjektif dan inkonsisten.
    
- **Distributor dan pengecer** dapat memastikan hanya apel berkualitas tinggi yang sampai ke konsumen, sehingga meningkatkan reputasi merek dan kepercayaan pelanggan.
    
- **Konsumen akhir** akan menerima produk dengan kualitas lebih konsisten, meningkatkan kepuasan dan loyalitas.
    
- **Perusahaan ekspor** dapat memenuhi standar mutu internasional dengan lebih mudah dan konsisten, sehingga memperluas pasar.
    

Implementasi sistem ini juga berpotensi menurunkan biaya operasional dan waktu proses sortasi hingga 20-30% (berdasarkan studi-studi otomasi kualitas di bidang agrikultur). Dengan demikian, proyek ini tidak hanya berdampak teknis, tetapi juga memberikan **nilai ekonomis dan kompetitif** bagi pelaku bisnis di bidang distribusi buah.
