# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # untuk array multidimensi

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)membuat dataframe, memudahkan dalam pengolahan data

import seaborn as sns #plotting grafik

import matplotlib.pyplot as plt #plotting grafik. menyajikan data ke bentuk visual

from sklearn.preprocessing import LabelEncoder #untuk labelencoder

from sklearn.model_selection import train_test_split #untuk membagi train dan test

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score #evaluasi model

import scikitplot as skplt #plotting grafik

#membuat model deep learning

import keras 

from keras.models import Sequential

from keras.layers import Dense



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

dataset = pd.read_csv("../input/adult.csv") #membaca data dan disimpan ke dalam variabel
dataset.head()
dataset.describe()#melihat statistika ringkasan data yang bersifat numerik
dataset.isnull()#untuk melihat data yang kosong atau null jika true
#data yang kosong sudah dirubah menjadi "?"

#removing '?' containing rows

dataset = dataset[(dataset != '?').all(axis=1)]

#label the income objects as 0 and 1

dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1})
dataset.head()
sns.catplot(x='education.num',y='income',data=dataset,kind='bar',height=6)

plt.show()

#melihat hubungan antara education.num dengan income dimana semakin tinggi education.num maka income juga akan semakin tinggi
sns.factorplot(x='marital.status',y='income',data=dataset,kind='bar',height=8)

plt.show()
sns.factorplot(x='relationship',y='income',data=dataset,kind='bar',size=7)

plt.show()
#reformat marital.status values to single and married

dataset['marital.status']=dataset['marital.status'].map({'Married-civ-spouse':'Married', 'Divorced':'Single', 'Never-married':'Single', 'Separated':'Single', 

'Widowed':'Single', 'Married-spouse-absent':'Married', 'Married-AF-spouse':'Married'})
dataset.head()
#setiap kolom di label encode menjadi angka sebagai penyesuaian terhadap model yang akan dibuat

for column in dataset:

    enc=LabelEncoder()

    if dataset.dtypes[column]==np.object:

         dataset[column]=enc.fit_transform(dataset[column])
plt.figure(figsize=(14,10))

sns.heatmap(dataset.corr(),annot=True,fmt='.2f')

plt.show()

#education dan education.num sangat berhubungan sehingga tidak dibutuhkan keduanya untuk menjelaskan data jadi hapus 

#salah satu attribut. begitu juga dengan marital.status dan relationship.

#nb jika heatmap>0.3 maka highly correlated
dataset=dataset.drop(['relationship','education'],axis=1)

dataset=dataset.drop(['occupation','fnlwgt','native.country'],axis=1)

#occupation di drop karena workclass dengan occupation mempunyai fungsi yang sama untuk menjelaskan status pekerjaan 

#sedangkan untuk fnlwft di drop karena tidak dibutuhkan untuk prediksi income dan native.country di drop karena dapat 

#menyebabkan bias dikarenakan banyak orang berasal dari satu negara 
dataset.head()
dataset.iloc[:,:]#ngetes mengambil semua baris dan kolom
X=dataset.iloc[:,0:-1] #mengambil atribut predictor dari kolom age ke hours per week 

y=dataset.iloc[:,-1]#mengambil kolom target income 

print(X.head())

print(y.head())

# Splitting the dataset into the Training set and Test set

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)

#x_train dan y_train untuk ngelatih model yang kita buat untuk memprediksi output

#x_test untuk menguji seberapa akurat model yang sudah ditrain

#y_test hasil sesungguhnya dari x_test digunakan untuk membandingkan hasil prediksi model terhadap x_test
classifier = Sequential()#inisialisasi awal ann
#input ada 9, hidden layer 1, 

#kernel_initializer = 'uniform' untuk membuat weight awal dekat dengan nol tapi bukan nol

#ReLU pada intinya hanya membuat pembatas pada bilangan nol, artinya apabila x ≤ 0 maka x = 0 dan apabila x > 0 maka x = x

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))
#membuat output layer 1 fungsi aktivasi sigmoid yang artinya hanya ada 0 dan 1 

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#optimizer adalah pilihan algoritma set yang digunakan untuk menentukan bobot/weight 

#adam adalah salah satu algoritma stochastic gradient descent 

#loss untuk memilih algoritma penentuan adjustment loss function, untuk kategori lebih dari 2 maka

#loss = categorical_crossentropy

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#batch_size : jumlah observasi kita mau update weights

#Epoch : jumlah keseluruhan training set untuk seluruh dataset atau bisa disebut iterasi

#mulai train

classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)
y_pred = classifier.predict(x_test) #membuat model memprediksi x_test yang hasilnya diletakkan ke y_pred

y_pred = (y_pred > 0.5) #mengambil angka yang apabila bernilai diatas 0.5 menjadi 1 

cm = confusion_matrix(y_test, y_pred) #membuat sebuah confusion matriks dari 

#target hasil prediksi dengan target sebenarnya

cm
print(accuracy_score(y_test,y_pred)*100) #output akurasi dari model yang telah dibuat 