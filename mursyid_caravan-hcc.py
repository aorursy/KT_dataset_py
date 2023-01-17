#Import semua library yang dibutuhkan 

import pandas as pd # Untuk memanipulasi file csv

import numpy as np # Untuk melakukan proses aritmetika pada matriks

import matplotlib.pyplot as plt # Untuk visualisasi pada dataset



# Membelah dataset menjadi train dan test, yang kedua untuk mencari hyperparameter yang baik

from sklearn.model_selection import train_test_split, GridSearchCV

# Model KNN nya sendiri

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

# Melakukan proses standardisasi pada suatu dataset 

from sklearn.preprocessing import StandardScaler

# Memberi label untuk tipe data kategorik

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

# Membantu dalam hal mengisi missing value

from sklearn.preprocessing import Imputer

# Mengukur akurasi model terhadap data yang diketahui

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# Import dataset ke Python

df = pd.read_csv('../input/caravan-insurance-challenge.csv')

df.head() # Melihat beberapa baris pertama dari dataset
x=df.iloc[:, :85]

y=df.iloc[:, 85]

x.head()
from sklearn.model_selection import train_test_split

X_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state = 0 )
# Loading library

from sklearn.neighbors import KNeighborsClassifier



# inisiasi Learning model ( k = 3 )

knn = KNeighborsClassifier( n_neighbors = 3 )



# fitting model tersebut

knn.fit( X_train, y_train )



# prediksi responnya

pred = knn.predict( x_test )



# evaluasi tingkat akurasi

# from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score

print ( accuracy_score( y_test, pred) )
# Buat list untuk menginisiasi nilai K, dari 1 sampai 49

list_k = list( range(1,50) )

print(list_k)



# membuat subset yang hanya memiliki nilai ganjil

neighbors = []

for x in list_k:

    if x%2 != 0:

        neighbors.append(x)

print( neighbors)



# list kosong yang akan menyimpan nilai dari cross-validation

cv_scores = []



# impor dulu cross_val_score

from sklearn.model_selection import cross_val_score



# lakukan 10-fold cross validation

for k in neighbors:

    knn = KNeighborsClassifier( n_neighbors = k )

    scores = cross_val_score( knn, X_train, y_train, cv = 10, scoring = 'accuracy' )

    cv_scores.append( scores.mean() )
# merubah nilai akurasi menjadi misklasifikasi atau error

MSE = [ 1-x for x in cv_scores ]



# Menentukan nilai k yang terbaik

k_optimal = neighbors[ MSE.index(min(MSE)) ]

print ("Nilai K yang paling optimal adalah %d" % k_optimal)

print ("K tersebut memiliki akurasi %lf" % cv_scores[ MSE.index(min(MSE)) ] )
