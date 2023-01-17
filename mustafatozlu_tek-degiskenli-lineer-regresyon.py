# Gerekli Tüm kütüphaneler import edildi.

import pandas as pd #csv formattaki veriyi almak için kullanacaz

import numpy as np # matematiksel işlemlerde hızlı hesaplamak için kullanacaz 

import matplotlib.pyplot as plt # grafik çizdirmek için kullanacaz

%matplotlib inline

plt.rcParams['figure.figsize'] = (8.0,6.0) # çıkan grafiğin boyutunu belirliyoruz

#Verilerin dataset üzerinden alınması

dataset = pd.read_csv('../input/SalaryData.csv')

dataset.head() #alınan veriler gösterilmesi için
dataset.describe() #Dataset dosyası hakkında istatistiksel olarak bilgi veriyor
#csv değerlerini ayırıyoruz

X = dataset.iloc[:,0] 

Y = dataset.iloc[:,1]

#Grafik üzerinde X ve Y leri koyuyoruz

plt.scatter(X,Y) 

plt.xlabel("deneyim")

plt.ylabel("maas")

plt.show()
#X ve Y için ayrı ayrı ortalamayı buluyoruz 

X_mean = np.mean(X)

Y_mean = np.mean(Y)



# Datanın modeli yapılıyor

lsmPay = 0

lsmPayda = 0

for i in range(len(X)):

    lsmPay += (X[i]- X_mean)*(Y[i] - Y_mean) #Last Square için pay

    lsmPayda += (X[i] - X_mean)**2 #Last Square için payda

m = lsmPay / lsmPayda #formuldeki Xin kat sayısı

c= Y_mean - m*X_mean # formuldeki sabit değişken



# değerlerin ekrana basılması

print('m=',m,'c=',c)

print('Modelimiz => Y=',m,'X +',c) 

print("Modele göre deneyim 2 iken alacagi maas: "+str((m*2)+(c) ))

print("Modele göre deneyim 5 sene iken alacagi maas: "+str((m*5)+(c) ))

#Tahmin Yapılacak

Y_pred = m*X +c



#MSE hesaplaması

summation = 0 #farkların toplamı tutacak değişken  

n = len(Y) #listedeki bulunan satır sayısı

for i in range (0,n):  

  difference = Y[i] - ((m*X[i])+(c))  #Gerçek değer ile tahmin edilen değer farkı

  squared_difference = difference**2  #farkın karesi alınır 

  summation = summation + squared_difference  #Tüm farklar toplanır

MSE = summation/n  #Tüm toplamlar eleman sayısına bölünür

print ("Mean Square Error değeri : " , MSE)



plt.scatter(X,Y) #Gerçek noktalar

plt.plot([min(X),max(X)],[min(Y_pred),max(Y_pred)],color='red') #Modelin tahmin ettiği çizim

#grafik isimleri

plt.xlabel("deneyim") 

plt.ylabel("maas")

plt.show()


