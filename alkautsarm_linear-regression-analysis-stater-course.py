'''

Import Library

Meng-import library yang akan dipakai sehingga perlu dilakukan import terlebih dahulu



Fungsi masing-masing libary

Scikit-Learn: Library untuk machine learning

Matplotlib: Library untuk melakukan visualisasi data

Numpy: Library untuk melakukan operasi matriks

Pandas: Library untuk memanipulasi data

Seaborn: Library untuk membuat grafis statistik

'''

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



'''

Import Dataset

'''

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
'''

Read Dataset

Membaca dataset, mengganti kolom untuk mempermudah, dan menampilkan 10 data pertama

'''



data = pd.read_csv('/kaggle/input/1000-cameras-dataset/camera_dataset.csv') #Membaca data

data.head(10) #Menampilkan 10 data teratas
'''

Column Name Changing

Merubah nama kolom untuk kemudahan dalam pemrograman

'''

data.columns = ['model', 'release_date', 'max_res', 'low_res', 'eff_pixels', 'zoom_wide', 'zoom_tele', 

               'normal_fr', 'macro_fr', 'storage', 'weight', 'dim', 'price'] #Mengganti nama-nama kolom

data.head(10) #Menampilkan 10 data teratas
'''

Handling Duplicate Values 

Mencari data yang sama atau duplikat

Menghapus data yang duplikat jika ditemukan

'''

result_data = data.drop_duplicates() #Melakukan drop data yang duplikat
'''

Handling Missing Values

- Mencari terlebih dahulu kolom yang memiliki data null

- Mereplace kolom yang memiliki data kosong dengan rata-rata dari kolom tersebut (mean imputation)

'''

print(result_data.isnull().sum()) #Menampilkan jumlah data null pada suatu kolom



result_data['macro_fr'].fillna(result_data['macro_fr'].mean(), inplace=True) #Mengisi data yang kosong di kolom tsb dengan rata-rata dari kolom tsb

result_data['storage'].fillna(result_data['storage'].mean(), inplace=True) #Mengisi data yang kosong di kolom tsb dengan rata-rata dari kolom tsb

result_data['weight'].fillna(result_data['weight'].mean(), inplace=True) #Mengisi data yang kosong di kolom tsb dengan rata-rata dari kolom tsb

result_data['dim'].fillna(result_data['dim'].mean(), inplace=True) #Mengisi data yang kosong di kolom tsb dengan rata-rata dari kolom tsb



print(result_data.isnull().sum()) #Menampilkan jumlah data null pada suatu kolom
'''

Describe Data

Melihat deskripsi data terutama dari masing-masing kolom

'''

result_data.describe() #Mendeskripsikan masing-masing kolom (count, mean, std, dan lain-lain)
'''

Column Selection

Pemilihan kolom untuk dianalisis dalam regression analysis

'''

final_data = result_data[['max_res','price']] #Memilih kolom yang akan dilakukan analisis regresi

final_data.describe() #Mendeskripsikan masing-masing kolom (count, mean, std, dan lain-lain)
'''

Visualisasi Data

Data dari masing-masing kolom divisualkan dalam 2 bentuk yang berbeda

'''

sns.countplot(final_data['max_res']) #Bentuk visualisasi bar chart

plt.show() #Menampilkan visualisasi



sns.countplot(final_data['price']) #Bentuk visualisasi bar chart

plt.show() #Menampilkan visualisasi



sns.distplot(final_data['max_res'], bins=10, kde=True) #Bentuk visualisasi histogram

plt.show() #Menampilkan visualisasi



sns.distplot(final_data['price'], bins=10, kde=True) #Bentuk visualisasi histogram

plt.show() #Menampilkan visualisasi
'''

Data Training & Testing

- Membagi training dan testing data dengan ukuran 80% training & 20% testing

- Melakukan training dan testing data

'''

x = final_data['max_res'].values.reshape(-1,1) #Dilakukan untuk mengubah menjadi 2D array dan -1,1 karena memiliki single feature

y = final_data['price'].values.reshape(-1,1) #Dilakukan untuk mengubah menjadi 2D array dan -1,1 karena memiliki single feature



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) #Membagi data menjadi data training dan testing



regressor = LinearRegression()

regressor.fit(x_train, y_train) #Menyocokkan model linier melalui training



y_pred = regressor.predict(x_test) #Melakukan prediksi
'''

Compare Actual & Predict Y

Membandingkan antara nilai actual dan predict dari Y

Memvisualisasikan perbandingan nilai actual dan predict dari Y

'''

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

compare_df = df.head(25) #Mengambil 25 data pertama



print(compare_df) #Menampilkan data yang dibandingkan (actual & result)



compare_df.plot(kind='bar') #Menampilkan data yang dibandingkan (actual & result) dalam bentuk diagram bar

plt.show() #Menampilkan visualisasi
'''

Data Visualization

Visualiasi data menggunakan scatter plot dan terdapat garis persamaan

'''



plt.scatter(x_test, y_test, color='blue') #Menampilkan visualiasi scatter dengan warna lingkaran biru

plt.plot(x_test, y_pred, color='red') #Menampilkan visualiasi garis persamaan regresi

plt.show() #Menampilkan visualisasi
'''

Final Regression Analysis

Analisis regresi yang meliputi mean absolute error, mean squared error, root mean squared error

Bagian ini juga memuat koefisien determinasi dan persamaan regresi

'''



print('Intercept:', regressor.intercept_[0]) #regressor.intercept_ == intercept == a

print('Slope:', regressor.coef_[0][0]) #regressor.coef_ == slope == b



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) #Menghitung MAE

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)) #Menghitung MSE

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #Menghitung RMSE

print('Coefficient of Determination:', metrics.r2_score(y_test, y_pred)) #Menghitung koefisien determinasi



print('Persamaan regresi: Y = {b:.3f}X + {a:.3f}'.format(b = regressor.coef_[0][0], a = regressor.intercept_[0])) #Menampilkan persamaan regresi