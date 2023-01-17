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
df = pd.read_csv("/kaggle/input/lokmaci/lokmaci.csv", encoding="latin1")



df.info()

df.head()
y = df.Oran.values.reshape(-1,1)

x = df.Aylar.values.reshape(-1,1)



plt.scatter(x,y)

plt.ylabel("Oran")

plt.xlabel("Aylar")

plt.show()
from sklearn.linear_model import LinearRegression



lr = LinearRegression()



lr.fit(x,y)



y_head = lr.predict(x)



plt.plot(x,y_head,color="red",label ="linear")

plt.title("Linear Regression Lokmacı Kazanç Tahmini")

plt.show()



print("2. sene tahmini: ",lr.predict([[24]]))
from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 3)



x_polynomial = polynomial_regression.fit_transform(x)



linear_regression2 = LinearRegression()

linear_regression2.fit(x_polynomial,y)



y_head2 = linear_regression2.predict(x_polynomial)



plt.plot(x,y_head2,color= "green",label = "Kazanç")

plt.title("Polynomial Regression Lokmacı Kazanç Tahmini")

plt.legend()

plt.show()



p = polynomial_regression.fit_transform([[15]])



print("2 sene sonra kazanç tahmini: ",linear_regression2.predict(p))
def riskOraniHesapla(y1,y2):    

    

    if y1 < y2:

        rate_text = "Düşük"

    else:

        rate_text = "Yüksek"

    

    return rate_text

    
#Mart ayına kadar olan datalar



df_mart = df.head(3)



y = df_mart.Oran.values.reshape(-1,1)

x = df_mart.Aylar.values.reshape(-1,1)



polynomial_regression = PolynomialFeatures(degree = 3)



x_polynomial = polynomial_regression.fit_transform(x)



linear_regression2 = LinearRegression()

linear_regression2.fit(x_polynomial,y)



y_head2 = linear_regression2.predict(x_polynomial)



p_nisan = polynomial_regression.fit_transform([[4]])

p_mayis = polynomial_regression.fit_transform([[5]])

p_haziran = polynomial_regression.fit_transform([[6]])



rate_nisan = riskOraniHesapla(y[2],linear_regression2.predict(p_nisan))

rate_mayis = riskOraniHesapla(y[2],linear_regression2.predict(p_mayis))

rate_haziran = riskOraniHesapla(y[2],linear_regression2.predict(p_haziran))



print("Mart ayına kadar veriler ile")

print("Nisan Ayı Riski: ",rate_nisan)

print("Mayıs Ayı Riski: ",rate_mayis)

print("Haziran Ayı Riski: ",rate_haziran)

#Nisan ayına kadar olan datalar



df_nisan = df.head(4)



y = df_nisan.Oran.values.reshape(-1,1)

x = df_nisan.Aylar.values.reshape(-1,1)



polynomial_regression = PolynomialFeatures(degree = 3)



x_polynomial = polynomial_regression.fit_transform(x)



linear_regression2 = LinearRegression()

linear_regression2.fit(x_polynomial,y)



y_head2 = linear_regression2.predict(x_polynomial)



plt.plot(x,y_head2,color= "green",label = "Kazanç")

plt.title("Polynomial Regression Lokmacı Kazanç Tahmini")

plt.legend()

plt.show()



p_mayis = polynomial_regression.fit_transform([[5]])

p_haziran = polynomial_regression.fit_transform([[6]])

p_temmuz = polynomial_regression.fit_transform([[7]])



rate_mayis = riskOraniHesapla(y[3],linear_regression2.predict(p_mayis))

rate_haziran = riskOraniHesapla(y[3],linear_regression2.predict(p_haziran))

rate_temmuz = riskOraniHesapla(y[3],linear_regression2.predict(p_temmuz))



print("Nisan ayına kadar veriler ile")

print("Mayıs Ayı Riski: ",rate_mayis)

print("Haziran Ayı Riski: ",rate_haziran)

print("Temmuz Ayı Riski: ",rate_temmuz)