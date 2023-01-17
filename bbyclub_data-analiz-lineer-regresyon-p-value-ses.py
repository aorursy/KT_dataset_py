# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings #disable warnings

warnings.filterwarnings('ignore') 



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#import data

data=pd.read_csv("../input/veriler.csv")

data.head()
#veri on isleme

boy=data[["boy"]]

boy_kilo=data[["boy","kilo"]]

print(boy)

print(boy_kilo)
#missingValues import data

data=pd.read_csv("../input/eksikveriler.csv")

data.info()
#missingValues import data

data=pd.read_csv("../input/eksikveriler.csv")

#Eksik verileri dolduracağız

from sklearn.preprocessing import Imputer

#missing_values ne ile ifade ediliyor. strategy ile eksik değerleri ortalama ile dolduracaz. axis= 0 ile kolon bazında ortalama al

imputer= Imputer(missing_values="NaN", strategy="mean", axis=0)

#ortalaması alınacak veriler

yas=data.iloc[:,3:4].values #yas sütununu aldık

imputer=imputer.fit(yas) #imputer fonksiyonunu yeni çektiğimiz yas değişkenine uyguladık

data["yas"]= imputer.transform(yas) #impute edilmiş veriyi data tablosuna transfer ettik

print(data)
#ulke kodlarını sayısal verilere çeviriyoruz. Bunu için Label Encoder kullanıyoruz

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder() #Label Encoder sınıfından bir nesne tanımlıyoruz

data.iloc[:,0]=le.fit_transform(data.iloc[:,0]) #ülke sütununu al ve çevirerek data verisini güncelle

print(data['ulke']) #Son halini yazdır
from sklearn.preprocessing import OneHotEncoder #Kolon bazlı dönüşüm yapar. İgili etiket 1 diğerleri 0 olark işaretlenir

ohe=OneHotEncoder()

ulke=data.iloc[:,0:1].values

ulke=ohe.fit_transform(ulke).toarray()

print(ulke)
#Dataframe, numpy array den farklı olarak bir index bilgisine sahiptir.

#ulke dizisini dataframe e çevirip bizdeki data ile birleştiriyoruz

print(type(ulke))

sonuc=pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us']) #ulkeleri dataFrame e çevir

print(sonuc)

sonuc2= pd.DataFrame(data=data.iloc[:,1:4], index=range(22),columns=["boy","kilo","yas"]) #boy yas kilo yu dataframe e cevir. Concat için gerekli

print(sonuc2)

cinsiyet=pd.DataFrame(data=data.iloc[:,-1:].values, index=range(22), columns=["cinsiyet"]) #cinsiyeti dataframe e çevir

print(cinsiyet)

sonuc3=pd.concat([sonuc,sonuc2],axis=1) #önce ilk iki sonuc birleştir

sonuc4=pd.concat([sonuc3,cinsiyet],axis=1) #tüm verileri tek data frame de birleştir

data=sonuc4

print(data)
#makinenin öğreneceği veri kümesi cinsiyetin olmadığı dataframe yani sonuc3 olacak

#bağımsız değişkenin alınacağı frame cinsiyet

#Rastgele bir dağılım yapıyoruz ki ezbe olmasın ya da öğrenme sadece belli veriler için geçerli olmasın. 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(sonuc3, cinsiyet,test_size=0.33)
#Nihai olarak yapmak istediğimiz 

#x_train' i kullanarak y_train' i öğren ve öğrendiklerinle x_test' i okuyarak y_test' i tahmin et!

print("X_train: \n",x_train)

print("------\n")

print("Y_train: \n",y_train)

print("------\n")

print("X_test: \n",x_test)

print("------\n")

print("Y_test: \n",y_test)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()



X_train=sc.fit_transform(x_train)

X_test=sc.fit_transform(x_test)
#Satışlar veri setini kullanacağız

data=pd.read_csv("../input/satislar.csv")

data.head()
#veri on isleme

aylar=data[['Aylar']]

print(aylar)

satislar=data[["Satislar"]] #ya da sat,slar= data.iloc[:,:1].values

print(satislar)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(aylar, satislar,test_size=0.33)

#Nihai olarak yapmak istediğimiz satislar tablosu için 

#x_train' i kullanarak y_train' i öğren ve öğrendiklerinle x_test' i okuyarak y_test' i tahmin et!

print("X_train: \n",x_train)

print("------\n")

print("Y_train: \n",y_train)

print("------\n")

print("X_test: \n",x_test)

print("------\n")

print("Y_test: \n",y_test)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()



X_train=sc.fit_transform(x_train)

X_test=sc.fit_transform(x_test)

Y_train=sc.fit_transform(y_train)

Y_test=sc.fit_transform(y_test)
#Lineer Regresyon model inşası standardize edilmiş değerlerle

from sklearn.linear_model import LinearRegression

lr= LinearRegression()

lr.fit(X_train, Y_train)

tahmin=lr.predict(X_test)

print(tahmin)

Y_test
#Lineer Regresyon model inşası normal edilmiş değerlerle

from sklearn.linear_model import LinearRegression

lr= LinearRegression()

lr.fit(x_train, y_train)

tahmin=lr.predict(x_test)

print(tahmin)

y_test
#Visualization

#indexlerine göre sırlama yapıyoruz ki plot doğru çıksın

x_train= x_train.sort_index()

y_train=y_train.sort_index()

plt.scatter(x_train,y_train) #şu an rastgele seçilmiş %66 lık veriyi çizdiriyoruz

plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara göre satış")

plt.xlabel("Aylar")

plt.ylabel("Satışlar")

print(lr.predict([[11]]))
data=pd.read_csv("../input/veriler.csv")
#cinsiyet kodlarını sayısal verilere çeviriyoruz. Bunu için Label Encoder kullanıyoruz



c=data.iloc[:,-1:].values

print(c)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder() #Label Encoder sınıfından bir nesne tanımlıyoruz

data.iloc[:,-1:]=le.fit_transform(data.iloc[:,-1:]) #cinsiyet sütununu al ve çevirerek data verisini güncelle

print(data['cinsiyet']) #Son halini yazdır



from sklearn.preprocessing import OneHotEncoder #Kolon bazlı dönüşüm yapar. İgili etiket 1 diğerleri 0 olark işaretlenir

ohe=OneHotEncoder(categorical_features="all")

c=ohe.fit_transform(c).toarray()

print("cinsiyet:",c)

cinsiyet=pd.DataFrame(data=c[:,:1], index=range(22), columns=["cinsiyet"]) #cinsiyeti dataframe e çevir. dummy variable' dan kurtul

print("cinsiyet dataframe:\n",cinsiyet)

print("sonuc4:\n",sonuc4)

sonuc4=sonuc4.iloc[:,:-1]

print("sonuc4:\n",sonuc4) #sonuc4 cinsiyetsiz sütun

sonuc5=pd.concat([sonuc4,cinsiyet],axis=1) #sonuc 4 ü cinsiyete kadar olan kısmını alıp yeni cinsiyetle birleştir

print(cinsiyet)

sonuc4.head()
sonuc5
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(sonuc4, cinsiyet,test_size=0.33)

#Nihai olarak yapmak istediğimiz satislar tablosu için 

#x_train' i kullanarak y_train' i öğren ve öğrendiklerinle x_test' i okuyarak y_test' i tahmin et!





from sklearn.preprocessing import StandardScaler

sc=StandardScaler()



X_train=sc.fit_transform(x_train)

X_test=sc.fit_transform(x_test)

Y_train=sc.fit_transform(y_train)

Y_test=sc.fit_transform(y_test)



from sklearn.linear_model import LinearRegression

regressor= LinearRegression()

regressor.fit(x_train, y_train)

tahmin=regressor.predict(x_test)

print(tahmin)

print(y_test)

print(y_train)
sonuc5
# boyu tahmin etmeye çalışalım. Bunun için boy sütununun solundaki ve sağındaki sütunları eğitim için kullanmak üzere birleştirip

# boy verisini yeni y_test olarak kullanacağız

#sonuc6 boy olmadan kullanacağımız data frame

boy=sonuc5.iloc[:,3:4].values

print(boy)

sol=sonuc5.iloc[:,:3] #Sol taraf boya kadar olan sütunlar

sag=sonuc5.iloc[:,4:] #sağ taraf boydan sonraki sütunlar

sonuc6=pd.concat([sol,sag],axis=1) #boy özelliği olmayan dataframe



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(sonuc6, boy,test_size=0.33)



from sklearn.linear_model import LinearRegression

reg2= LinearRegression()

reg2.fit(x_train, y_train)

tahmin=reg2.predict(x_test)

print("tahmin: \n",tahmin)

print("y_test değerleri: \n",y_test)
#Geri Eleme (Back Elimiation)

# Boy tahmini yaparken tahmin üzerinde en az etkiye sahip özellikleri (kolonları) çıkarmak istiyoruz



import statsmodels.regression.linear_model as sm

X= np.append(arr=np.ones((22,1)).astype(int),values=sonuc6, axis=1)

X_l=sonuc6.iloc[:,[0,1,2,3,4,5]].values

r_ols=sm.OLS(endog=boy, exog=X_l)

r=r_ols.fit()

print(r.summary())
#Sırayla P-value si en yüksek olan kolonları çıkarıyoruz

X_l=sonuc6.iloc[:,[0,1,2,3,5]].values

r_ols=sm.OLS(endog=boy, exog=X_l)

r=r_ols.fit()

print(r.summary())
#Sırayla P-value si en yüksek olan kolonları çıkarıyoruz

X_l=sonuc6.iloc[:,[0,1,2,3]].values

r_ols=sm.OLS(endog=boy, exog=X_l)

r=r_ols.fit()

print(r.summary())
#şimdi modelimizi tekrar test ediyoruz. Backward elimination yöntemini kullandık. Sistemi test ediyoruz yeni verilere göre

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X_l, boy,test_size=0.33)



from sklearn.linear_model import LinearRegression

reg3= LinearRegression()

reg3.fit(x_train, y_train)

tahmin=reg3.predict(x_test)

print("tahmin: \n",tahmin)

print("y_test değerleri: \n",y_test)
odev_data=pd.read_csv("../input/odev_tenis.csv")

odev_data.head()
#play, windy, outlook değerlerinii encodin yapacaz. play ve windiy 0,1 değerleri olduğu için dummy

#variable tuzağına düşmemek için label encoding, outlook için 3 farklı değer olduğundan one hot encoding yapacaz

#Bunu için kısayol kullanıyoruz. Şöyleki:

odev_data2=odev_data.apply(LabelEncoder().fit_transform) #tüm verileri label encoding yaptım. 

#Ancak hava durumu verilerini one hot encoding yapacam

o= odev_data2.iloc[:,:1]

ohe= OneHotEncoder(categorical_features="all")

o=ohe.fit_transform(o).toarray()

print(o)



outlook=pd.DataFrame(data=o,index=range(14), columns=['o','r','s'])

sonveriler= pd.concat([outlook,odev_data.iloc[:,1:3]],axis=1)

sonveriler=pd.concat([sonveriler,odev_data2.iloc[:,-2:]],axis=1)

sonveriler #preprocessing işlemi tamamlanmış oldu.                    
#humidity kolonunu test için ayırıyoruz

h_sol=sonveriler.iloc[:,0:4]

h_sag=sonveriler.iloc[:,-2:]

humidity=sonveriler.iloc[:,4:5]

x_l=pd.concat([h_sol,h_sag],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_l,humidity,test_size=0.33)



from sklearn.linear_model import LinearRegression

reg4= LinearRegression()

reg4.fit(x_train, y_train)

tahmin=reg4.predict(x_test)

print("tahmin: \n",tahmin)

print("y_test değerleri: \n",y_test)
import statsmodels.regression.linear_model as sm

X= np.append(arr=np.ones((14,1)).astype(int),values=x_l, axis=1)

x_l

X_l=x_l.iloc[:,[0,1,2,3,4,5]].values

r_ols=sm.OLS(endog=humidity, exog=X_l)

r=r_ols.fit()

print(r.summary())
#en yüksek p value ye sahip kolon x5. Onu atıyoruz.

import statsmodels.regression.linear_model as sm

X= np.append(arr=np.ones((14,1)).astype(int),values=x_l, axis=1)

x_l

X_l=x_l.iloc[:,[0,1,2,3,5]].values

r_ols=sm.OLS(endog=humidity, exog=X_l)

r=r_ols.fit()

print(r.summary())
#windy kolonunu çıkartıp tekrar prediction yapacam

x_train=pd.concat([x_train.iloc[:,0:4],x_train.iloc[:,-1]],axis=1)

x_train

x_test=pd.concat([x_test.iloc[:,0:4],x_test.iloc[:,-1]],axis=1)

x_test

#backward elimination sonrası tekrar tahmin ediyoruz

reg5= LinearRegression()

reg5.fit(x_train, y_train)

tahmin2=reg5.predict(x_test)

print("tahmin: \n",tahmin)

print("tahmin2: \n",tahmin2)

print("y_test değerleri: \n",y_test)



#Hemen hemen tüm sonuçlarda düzelme var!