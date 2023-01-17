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
data =pd.read_csv('/kaggle/input/calcofi/bottle.csv')

Salinity = data[['Salnty']]

Temperature = data[['T_degC']]

Salinity.fillna(Salinity.mean(), inplace=True)

Temperature.fillna(Temperature.mean(), inplace=True)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(Salinity,Temperature,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression

#Sınıftan bir nesne oluşturuyoruz.

lr = LinearRegression()



# Train veri kümelerini vererek makineyi eğitiyoruz.

lr.fit(x_train,y_train)
import matplotlib.pyplot as plt

# Aylar'ın test kümesini vererek Satislar'ı tahmin etmesini sağlıyoruz. Üst satırda makinemizi eğitmiştik.

tahmin = lr.predict(x_test)





# Verileri grafikte düzenli göstermek için index numaralarına göre sıralıyoruz.

x_train = x_train.sort_index()

y_train = y_train.sort_index()



# Grafik şeklinde ekrana basıyoruz.

plt.plot(x_train,y_train)

plt.plot(x_test,tahmin)

plt.xlabel('Salinity')

plt.ylabel('Temperature')

plt.show()