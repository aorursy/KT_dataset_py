# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as seabornInstance

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics



#Load data dari dataset Insurance



dataset = pd.read_csv('/kaggle/input/insurance/insurance.csv')

dataset.shape



#Disini saya memakai metode Simple Linear Regression



print(dataset)



#Memperlihatkan data dari dataset yang saya pakai untuk analisis.

#Disini saya memakai data Umur dan Charges untuk membandingkan dan menganalisa hubungan umur dengan harga tanggungan
#Plot atau view data ke dalam grafik untuk visualisasi. 

#Dikarenakan pola data cenderung linier, saya memodelkannya ke dalam model persamaan linier



dataset.plot(x='age', y='charges', style='o')

plt.title( 'Age and Charge')

plt.xlabel('Age')

plt.ylabel('Charge')



# dengan ini, Memperlihatkan visualisasi Hubungan antara Umur seseorang dengan harga yang telah ditentukan untuk pembayaran Tanggungan kesehatannya
# Memetakan dataset ke variabel X dan Y



X = dataset['age'].values.reshape(-1,1)

y = dataset['charges'].values.reshape(-1,1)



#Memilah/split dataset menjadi 80 % untuk data training dan sisanya untuk data testing. 

#test_size=0.2 artinya menentukan ukuran data test sebesar 0.2 atau 20 % dari dataset asli



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



#Di sini,memakai class dari library sklearn yaitu LinearRegression



regressor = LinearRegression()



#Berikutnya adalah training atau melatih data dari variabel X_train dan y_train. 



regressor.fit(X_train, y_train)

print(regressor.intercept_)

print(regressor.coef_)



#Secara matematis, kita sudah dapat model formula dari persamaan linier antara Umur seseorang dengan tanggungan yg harus dibayarkan.



y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test,  color='gray')

plt.plot(X_test, y_pred, color='red', linewidth=2)



#dari model, kita melakukan uji coba testing dengan sisa data.