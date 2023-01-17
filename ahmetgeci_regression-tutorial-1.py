#numpy bizim matematiksel kütüphanemiz

#pandas datamızı çekmek için kullanılır



import numpy as np 

import pandas as pd 





#datamızı  'data' değişkenine tanımla

data = pd.read_csv('../input/samplelineerregression/linear-regression-dataset.csv' ,  sep = ';')
#verisetimdeki sütünların isimleri

data.columns

#deneyim ve maas olarak 2 feature(öznitelik) var 
data.head(5)

# head() metodum verisetimdeki ilk baştan 5 satırı görmemi sağlar (head ' e ' paremetre göndermez iseniz 5 olarak otomatik atama yapar) 
data.tail(2)

# tail() sondan 2 satırmı listeler
data

#bütün verimizi listeler
#verisetimin satır ve sütün sayısı

data.shape

#14 satır 2 sütün bulunmakta
#datamızı x ve y olarak bölelim 

# x değişkenimizde : eğitmek istediğim verilerin Feature(öznitelikleri) bulunduracak.

# y değişkenimizde tanımlı olanlar ise bizim Target(hedef) bulunacaktır

#tek bir feature olduğu için sadece data.deneyim kullanabilirim

x= data.deneyim.values.reshape(-1,1)

y=data.maas.values.reshape(-1,1)
#grafiklerimi çizdirmek için matplotlib kütüphanemi ekliyorum 

import  matplotlib.pyplot as plt 



plt.scatter(x,y,color = 'blue')

plt.xlabel('Deneyim')

plt.ylabel('Maaş')

plt.show()

#sklearn machine learning algoritmalarımı kullanmamı sağlayan kütüphanedir.

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()



# fit metodu eğitimi başlatmak için kullanılır

# linear regression için 2 parametre gönderirdik 

#ilk eğitilecek olan fuature lar 

#ikinci eğitim sonucu target(hedefler)

linear_reg.fit(x,y)
#eğittimizi modelimizi test için tahminler yapalım 

y_head = linear_reg.predict(x)

#verisetimi tekrar tahmin edelim

y_head
#modelimizin ne kadar iyi performansta olduğuna bakalım



from sklearn.metrics import r2_score

#önce gerçek sonuçlarımı(Targets -> y) daha sonra modelimizin tahminlerini(y_head) gönderdik

print('lineer regression score : ' , r2_score(y,y_head))

#modelimizi grafikte çizdirelim

#datam 

plt.scatter(x,y,color='blue')

plt.xlabel('Deneyim')

plt.ylabel('Maas')

#modelim

plt.plot(x,y_head,color='red')

#ekrana yazdır 

plt.show()
multiple_Data = pd.read_csv('../input/samplemultipleregression/multiple-linear-regression-dataset.csv' , sep = ';')



#iloc x değişkenime birden fazla sütünu eklemek için kullanılır

#iloc ' un ' içine sütünların indexleri yazılır

# 0 -> deneyim   1-> maas   2-> yas 

multi_x = multiple_Data.iloc[:,[0,2]].values.reshape(-1,1)

multi_y =multiple_Data.maas.values.reshape(-1,1)





multi_x                            
from sklearn.linear_model import LinearRegression



multi_reg = LinearRegression()

multi_reg.fit(x,y)



multi_y_head =multi_reg.predict(multi_x)
poly_data = pd.read_csv('../input/samplepolynomialregression/polynomial-regression.csv',sep=';')
poly_y = poly_data.araba_max_hiz.values.reshape(-1,1)

poly_x = poly_data.araba_fiyat.values.reshape(-1,1)



lr = LinearRegression()

#linear regressionumu tekrar çağırıyorum
lr.fit(poly_x,poly_y)





y_head = lr.predict(poly_x)



plt.scatter(poly_x,poly_y)

plt.xlabel('fiyat')

plt.ylabel('max hız ')

plt.plot(poly_x,y_head, color = 'red')

plt.show()



#print(lr.predict(10000))

#linear regression modelimiz bu veriseti için uygun değil 

#o zaman modelimizi evrimleştirip polinoma dönüştürelim
from sklearn.preprocessing import PolynomialFeatures

#degree = derece denemektir

poly = PolynomialFeatures(degree = 4 )

poly_x_trans = poly.fit_transform(poly_x)


lr.fit(poly_x_trans,poly_y)

plt.scatter(poly_x,poly_y,color='orange')

plt.plot(poly_x , lr.predict(poly.fit_transform(poly_x)), color = 'blue')

plt.show()


