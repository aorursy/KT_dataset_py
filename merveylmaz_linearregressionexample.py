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
#Kütüphaneleri Tanımlama

from math import sqrt

from math import fabs

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

plt.rcParams['figure.figsize'] = (12.0, 9.0)



# Giriş Dataset Yükleme

data = pd.read_csv('/kaggle/input/salary-data/Salary_Data.csv') # Dateset i oku

X = data.iloc[:, 0] # Dataset in 0. sütunundaki değerleri oku ve X dizisine at

Y = data.iloc[:, 1] # Dataset in 1. sütunundaki değerleri oku ve Y dizisine at



# İlk ve İkinci öznitelik için scatter grafiğini çiz

plt.scatter(X, Y)

plt.xlabel('X Değerleri')

plt.ylabel('Y Değerleri')

plt.show() 

print(" Boyut : ",data.shape)



# Ortalamaların Hesaplanması

X_mean = np.mean(X) # Tüm X değerlerinin ortalamasını hesapla

Y_mean = np.mean(Y) # Tüm Y değerlerinin ortalamasını hesapla



num = 0

den = 0

# Y = mX + c -> X ve Y arasındaki doğrusal ilişki

for i in range(len(X)):

    num += (X[i] - X_mean)*(Y[i] - Y_mean) # Tüm X ve Y değerleri ile ortalama X ve Y değerleri ile 'num' hesapla

    den += (X[i] - X_mean)**2 #Tüm X değerleri ve ortalama X değeri ile 'den' hesapla



# den ve num dğerleri m ve c değerlerinin bulunmasında kullanılacak

m = num / den # m -> slope (eğim) hesapla

c = Y_mean - m*X_mean # c -> intercept hesapla

print (" m = ", m,"c = ", c) # Bulunan m ve c değerlerinin yazdır



# Standart Sapma Hesaplama

# X Değerleri İçin:

s = 1/len(X)-1

ss = sqrt(fabs(s*den))

print(" X Değerleri İçin Standart Sapma = ", ss)



# Y Değerleri İçin

for i in range(len(Y)):

    den += (Y[i] - Y_mean)**2 #Tüm Y değerleri ve ortalama Y değeri ile den hesapla

s = 1/len(Y)-1

ss = sqrt(fabs(s*den))

print(" Y Değerleri İçin Standart Sapma = ", ss)



# Tahmin Yapma

Y_pred = m*X + c # Tahmini Y değeri doğrusunu hesapla

plt.scatter(X, Y) # Gerçek X ve Y değerlerinin göster

plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # Tahmin edilen değerleri göster

plt.xlabel('X Değerleri')

plt.ylabel('Y Değerleri')

plt.show()



# Accuracy Hesaplama

lr = LinearRegression() # LinearRegression sınıfından bir nesne oluşturuyoruz.

df1 = pd.DataFrame(X) # X değerlerini frame e dönüştür

df2 = pd.DataFrame(Y) # Y değerlerini frame e dönüştür

lr.fit(df1,df2) # Veri kümelerini vererek makineyi eğitiyoruz. Kullanılan veriler hem eğitim hem de test için aynıdır.

accuracy = lr.score(df1, df2) # Accuracy hesapla

print(" Accuracy : ", accuracy)



#Gerçek değerle tahmin arasındaki benzerliğe göre mean absolute error, mean squared error ve root mean squared error hesaplama

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



mae = mean_absolute_error(lr.predict(df1), df2)

mse = mean_squared_error(lr.predict(df1), df2)

rmse = np.sqrt(mse)



print(' Mean Absolute Error (MAE): %.2f' % mae)

print(' Mean Squared Error (MSE): %.2f' % mse)

print(' Root Mean Squared Error (RMSE): %.2f' % rmse)