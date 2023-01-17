# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Kütüphane Tanımlama

import numpy as np                  

import matplotlib.pyplot as plt

import pandas as pd



# Dataset Okuma

dataset = pd.read_csv("../input/iris-dataset/iris.data.csv")

dataset.head()
# Dataset Train ve Test Olarak Ayırma

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 4].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
# Ölçekleme İşlemi

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

print(scaler.fit(X_train))



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
# Eğitim ve Tahmin İşlemleri

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(" Accuracy : ",accuracy_score(y_test, y_pred))
# Confusion Matrix ve Diğer Skorlar ile Algoritmanın Değelendirilmesi

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
# k Değerlerine Göre Hata Tahmini Oranı Hesaplama

error = []

# 1 ile 40 arasındaki k değerleri için hata hesaplama

for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error.append(np.mean(pred_i != y_test))

    

# Bulunan hata miktarlarının plt üzerinde gösterimi 

plt.figure(figsize=(12, 6))

plt.plot(range(1, 40), error, color='black', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('k Değeri için Error Oranı')

plt.xlabel('k Değeri')

plt.ylabel('Mean Error')
# k Değerlerine Göre Accuracy Hesaplama

neighbors = np.arange(1, 10) 

train_accuracy = np.empty(len(neighbors)) 

test_accuracy = np.empty(len(neighbors)) 

  

# 1 ile 10 arasındaki k değerleri için accuracy hesaplama

for i, k in enumerate(neighbors): 

    knn = KNeighborsClassifier(n_neighbors=k) 

    knn.fit(X_train, y_train) 

      

    # train ve test için accuracy hesaplama

    train_accuracy[i] = knn.score(X_train, y_train) 

    test_accuracy[i] = knn.score(X_test, y_test) 



# Hesaplanan değerlerin plt üzerinde gösterimi

plt.plot(neighbors, test_accuracy, marker='o', label = 'Testing dataset Accuracy') 

plt.plot(neighbors, train_accuracy, marker='*', label = 'Training dataset Accuracy') 

  

plt.legend() 

plt.xlabel('Komşu Sayısı (k)') 

plt.ylabel('Accuracy') 

plt.show() 