# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import warnings

import warnings

# ignore warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dic = {"deneyim":[0.5,0,1,5,8,4,15,7,3,2,12,10,14,6],

      "maas":[2500,2250,2750,8000,9000,6900,20000,8500,6000,3500,15000,13000,18000,7500]}
df = pd.DataFrame(dic)

df
plt.scatter(x=df.deneyim,y=df.maas)

plt.xlabel("Deneyim")

plt.ylabel("Maas")

plt.show()
# sklearn kullanıcaz

from sklearn.linear_model import LinearRegression



linear_reg = LinearRegression()



# x = df.deneyim # type(x) ----> pandas.core.series.Series

# Numpy kullanıcam. df.deneyim.values --> numpy çevirdik...

# x.shape # (14,) linear regression için (14,1) şekline getirmeliyim yoksa anlamaz.



x = df.deneyim.values.reshape(-1,1) # type(x) ----> numpy.ndarray

y = df.maas.values.reshape(-1,1) # y.shape(14,1)



linear_reg.fit(x,y) # bana bir line fit et
# prediction (tahmin)

b0 = linear_reg.predict([[0]])

print("b0 = ",b0)

# line'nın y eksenini kestiği yer  intercept



b0_ = linear_reg.intercept_

print("b0_ = ",b0_)



b1 = linear_reg.coef_

print("b1 = ",b1) # eğim   slope  b1



# maas = 1663 + 1138*deneyim  formülümüz

# istediğimiz değeri predict edebiliriz.
maas_yeni = 1663 + 1138*11

print(maas_yeni)



print(linear_reg.predict([[11]]))

# 11 yıllık deneyimi olanların maaşı 14181



linear_reg.predict([[13]]) # 13 yıllık maas tahmin ettik
# visualize line  ----> line görselleştircez.

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  

# deneyimimi yani x değerlerim



plt.scatter(x,y)



y_head=linear_reg.predict(array) # maas

plt.plot(array, y_head, color="red")



plt.show()



print("100 yıllık deneyim = maası",linear_reg.predict([[100]]))
y_head
dic["yas"] = [22,21,23,25,28,23,35,29,22,23,32,30,34,27]

df = pd.DataFrame(dic)

df
from sklearn.linear_model import LinearRegression



x = df.iloc[:,[0,2]].values # deneyim ve yas

y = df.maas.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()

multiple_linear_regression.fit(x,y) # bana bir line fit et
print("b0:",multiple_linear_regression.intercept_)

print("b1,b2:", multiple_linear_regression.coef_)
# predict

multiple_linear_regression.predict(np.array([[10,35],[5,35]]))

# yaşlar aynı ama deneyim farkı ile maas değiştiriyor.
dic = {"araba_fiyat":[60,70,80,100,120,150,200,250,300,400,500,750,1000,2000,3000],

      "araba_max_hiz":[180,180,200,200,200,220,240,240,300,350,350,360,365,365,365]}



df = pd.DataFrame(dic)
x = df.araba_fiyat.values.reshape(-1,1)

y = df.araba_max_hiz.values.reshape(-1,1)



plt.scatter(x,y)

plt.ylabel("araba_max_hiz")

plt.xlabel("araba_fiyat")

plt.show()
# linear regression =  y = b0 + b1*x

# multiple linear regression   y = b0 + b1*x1 + b2*x2



from sklearn.linear_model import LinearRegression



lr = LinearRegression()

lr.fit(x,y) # en uygun line'ı fit ediyoruz.



y_head = lr.predict(x) # her bir değere göre tahmin(predict) yapıyotuz.



y_head
plt.scatter(x,y)

plt.plot(x,y_head,"r")

plt.show()



# linear regression modeli fakat pek doğru değil
print("10 milyon tl lik araba hizi tahmini: ",lr.predict([[10000]]))
# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n

# x^2 elde etmeliyiz.



from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 2)

# degree(n) = 2  ---->  x^2'ye kadar



x_polynomial = polynomial_regression.fit_transform(x)

x_polynomial
# fit

linear_regression2 = LinearRegression()

linear_regression2.fit(x_polynomial,y)



y_head2 = linear_regression2.predict(x_polynomial)



plt.scatter(x,y)

plt.plot(x,y_head,color="red", label = "linear")

plt.plot(x,y_head2,color= "green",label = "poly")

plt.legend()

plt.show()



# NOT = degree arttırırsak daha da iyileştirebiliriz. degree = 4 dene.
dic = {"seviye":[1,2,3,4,5,6,7,8,9,10],

      "fiyat":[100,80,70,60,50,40,30,20,10,5]}



df = pd.DataFrame(dic)
x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,1].values.reshape(-1,1)
# decision tree regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor() # random state = 0 ??

tree_reg.fit(x,y)



# ağaç yapısını oluşturduk
y_head = tree_reg.predict(x)



tree_reg.predict([[5.5]])
# visualize - görselleştirme



plt.scatter(x,y,color = "red")

plt.plot(x,y_head,color = "green")

plt.xlabel("Tribun level")

plt.ylabel("ucret")

plt.show()



# Oluşan grafik doğru değil çünkü bir leaf bulunan değerler aynı olmalı burda farklı farklı

# örneğin 100 ile 80 arasında azalmış ama sabit olmalıydı
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = tree_reg.predict(x_)
plt.scatter(x,y,color = "red")

plt.plot(x_,y_head,color = "green")

plt.xlabel("Tribun level")

plt.ylabel("ucret")

plt.show()
dic = {"seviye":[1,2,3,4,5,6,7,8,9,10],

      "fiyat":[100,80,70,60,50,40,30,20,10,5]}



df = pd.DataFrame(dic)



# df = pd.read_csv("random_forest_regression_dataset.csv",sep=";",header=None)
x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,1].values.reshape(-1,1)
x
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)



# n_estimators 100 ağaç kullanıcam

# random_state yapılan rastgele seçimi aynı olmasını sağlar. 

# Yani algoritmayı iki kez çalıştırırsak 1. ve 2. sonuçlar farklı olabilir çünkü rastgele seçim var bunu engelliyoruz.Hep aynı seçilde seçim yaptırıyoruz.

# Aynı random değerlerinin seçilmesini sağlar.



rf.fit(x,y)
print("7.8 seviyesinde fiyatın ne kadar olduğu = ", rf.predict([[7.8]]))
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = rf.predict(x_)
# görselleştirme



plt.scatter(x,y,color="red")

plt.plot(x_,y_head,color="green")

plt.xlabel("Tribun level")

plt.ylabel("ucret")

plt.show()



# Decision tree den farklı olarak 100 ağaç kulladık ve sonuçlar daha iyi çıktı.
dic = {"seviye":[1,2,3,4,5,6,7,8,9,10],

      "fiyat":[100,80,70,60,50,40,30,20,10,5]}



df = pd.DataFrame(dic)



x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,1].values.reshape(-1,1)



from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)



rf.fit(x,y)
y_head = rf.predict(x)



y_head
from sklearn.metrics import r2_score

print("r_score: ", r2_score(y,y_head))



# 1'e yakın, y'nin üzerinden predict (y_head) ettiğimiz değerlerin r-square değeri
dic = {"deneyim":[0.5,0,1,5,8,4,15,7,3,2,12,10,14,6],

      "maas":[2500,2250,2750,8000,9000,6900,20000,8500,6000,3500,15000,13000,18000,7500]}

df = pd.DataFrame(dic)



from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()



x = df.deneyim.values.reshape(-1,1)

y = df.maas.values.reshape(-1,1)



linear_reg.fit(x,y)



y_head = linear_reg.predict(x)



plt.scatter(x,y)

plt.plot(x,y_head,color="red")
from sklearn.metrics import r2_score



print("r_source: ",r2_score(y,y_head))



# fark fazla olmasada random forest 1'e daha yakın

# r-square with random forest , r_source: 0.9798724794092587