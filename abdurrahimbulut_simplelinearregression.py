# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
veriler = pd.read_csv('/kaggle/input/salesbymonths/satislar.csv')
aylar = veriler[['Aylar']]

#print(aylar)

satislar = veriler[['Satislar']]

#print(satislar)
#verilerin egitim ve test icin bolunmesi

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

#print(x_train)

#print(x_test)

#print(y_train)

#print(y_test)
#verilerin olceklenmesi

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(x_train)

X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)

Y_test = sc.fit_transform(y_test)
#linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,Y_train)
tahmin = lr.predict(X_test)

print("Tahmin (Ölçekli,Scale)")

print(tahmin)

print("*********")

print("Gerçek veri (Ölçekli,Scale)")

print(Y_test)
#ÖLÇEKLENMEMİŞ VERİLER İÇİN (unscaled data)

#linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)

print("Tahmin")

print(tahmin)

print("*********")

print("Gerçek veri")

print(y_test)
x_train = x_train.sort_index()

y_train = y_train.sort_index()
plt.plot(x_train,y_train)

plt.plot(x_test,lr.predict(x_test))

plt.title("aylara göre satış")

plt.xlabel("Aylar")

plt.ylabel("Satışlar")