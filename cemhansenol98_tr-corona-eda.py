# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#import visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Toplam_Vaka_Sayisi = Total Infected Values

#Toplam_Olum_Sayisi = Total Death Values

#Gunluk_Vaka_Sayisi = Daily Infected Values

#Gunluk_Olu_Sayisi = Daily Death Values
df = pd.read_csv('/kaggle/input/tr-corona-dataset/coronatrv3.csv', sep=';')

df
df.info()
df.describe()
chart = sns.barplot(x="Tarih",y="Gunluk_Vaka_Sayisi",data = df)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.title("Tarih - Gunluk Vaka Sayisi")
chart = sns.barplot(x="Tarih",y="Gunluk_Olu_Sayisi",data = df)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.title("Tarih - Gunluk Olum Sayisi")
f,chart = plt.subplots(figsize = (9,15))

sns.barplot(x=df['Tarih'],y=df['Gunluk_Vaka_Sayisi'],color='green',alpha = 0.5,label='Gunluk Vaka Sayisi')

sns.barplot(x=df['Tarih'],y=df['Gunluk_Olu_Sayisi'],color='red',alpha = 0.7,label='Gunluk Olum Sayisi')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.legend(loc='upper left')

plt.title("Tarih - Gunluk Vaka ve Gunluk Olum Sayisi")
f,chart = plt.subplots(figsize = (9,15))

sns.barplot(x=df['Tarih'],y=df['Toplam_Vaka_Sayisi'],color='green',alpha = 0.5,label='Toplam Vaka Sayisi')

sns.barplot(x=df['Tarih'],y=df['Toplam_Olum_Sayisi'],color='red',alpha = 0.7,label='Toplam Olum Sayisi')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.legend(loc='upper left')

plt.title("Tarih - Toplam Vaka ve Toplam Olum Sayisi")
sns.pointplot(x=df['Tarih'],y=df['Toplam_Vaka_Sayisi'],data=df,color='green',alpha=0.8)

plt.xticks(rotation=45)
sns.pointplot(x=df['Tarih'],y=df['Toplam_Olum_Sayisi'],data=df,color='red',alpha=0.8)

plt.xticks(rotation=45)
sns.pointplot(x=df['Tarih'],y=df['Gunluk_Vaka_Sayisi'],data=df,color='green',alpha=0.8)

plt.xticks(rotation=45)
sns.pointplot(x=df['Tarih'],y=df['Gunluk_Olu_Sayisi'],data=df,color='red',alpha=0.8)

plt.xticks(rotation=45)
#Polynomial Regression with degree = 2

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

x=df.Toplam_Vaka_Sayisi.values.reshape(-1,1)

y=df.Toplam_Olum_Sayisi.values.reshape(-1,1)



polynomial_regression = PolynomialFeatures(degree=2)

x_polynomial = polynomial_regression.fit_transform(x)



linear_regression = LinearRegression()

linear_regression.fit(x_polynomial,y)



y_head = linear_regression.predict(x_polynomial)



plt.plot(x,y_head,color="green")

plt.scatter(x,y)

plt.xlabel('Toplam Vaka Sayisi')

plt.ylabel('Toplam Olum Sayisi')

plt.title("Degree = 2 iken Regression Dogrusu")

plt.legend()

plt.show()
#Polynomial Regression with degree = 4

#degree 4 olunca overfit oluyor gibi..

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

x=df.Toplam_Vaka_Sayisi.values.reshape(-1,1)

y=df.Toplam_Olum_Sayisi.values.reshape(-1,1)



polynomial_regression = PolynomialFeatures(degree=4)

x_polynomial = polynomial_regression.fit_transform(x)



linear_regression = LinearRegression()

linear_regression.fit(x_polynomial,y)



y_head = linear_regression.predict(x_polynomial)



plt.plot(x,y_head,color="green",label="Tahmin Dogrusu")

plt.scatter(x,y)

plt.xlabel('Toplam Vaka Sayisi')

plt.ylabel('Toplam Olum Sayisi')

plt.title("Degree = 4 iken Regression Dogrusu")

plt.legend()

plt.show()
#Polynomial Regression with degree = 2

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

x=df.Gunluk_Vaka_Sayisi.values.reshape(-1,1)

y=df.Gunluk_Olu_Sayisi.values.reshape(-1,1)



polynomial_regression = PolynomialFeatures(degree=2)

x_polynomial = polynomial_regression.fit_transform(x)



linear_regression = LinearRegression()

linear_regression.fit(x_polynomial,y)



y_head = linear_regression.predict(x_polynomial)



plt.plot(x,y_head,color="green")

plt.scatter(x,y)

plt.xlabel('Gunluk Vaka Sayisi')

plt.ylabel('Gunluk Olum Sayisi')

plt.title("Degree = 2 iken Regression Dogrusu")

plt.legend()

plt.show()
#Polynomial Regression with degree = 4

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

x=df.Gunluk_Vaka_Sayisi.values.reshape(-1,1)

y=df.Gunluk_Olu_Sayisi.values.reshape(-1,1)



polynomial_regression = PolynomialFeatures(degree=4)

x_polynomial = polynomial_regression.fit_transform(x)



linear_regression = LinearRegression()

linear_regression.fit(x_polynomial,y)



y_head = linear_regression.predict(x_polynomial)



plt.plot(x,y_head,color="green")

plt.scatter(x,y)

plt.xlabel('Gunluk Vaka Sayisi')

plt.ylabel('Gunluk Olum Sayisi')

plt.title("Degree = 4 iken Regression Dogrusu")

plt.legend()

plt.show()