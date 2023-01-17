# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

#sk-learn
from sklearn.preprocessing import Imputer 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

#WordCloud
from wordcloud import WordCloud

import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/countries of the world.csv")
data.head()  #dataya kısaca göz atmamız için bize ilk 5 satırı görüntüler.
data.tail(10) #datanın farklı satırlarını görüntülemek için bize son 5 satırı görüntüler.
data.info()  #data featureları hakkında bize detaylı bilgi verir.
             #featurelardaki kayıt sayılarını verir.
             #Bellek kullanımı hakkında bilgiler verir.
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('.', '').str.replace('%','').str.replace('/','')
#float ve/veya int tipine çevirebilmek için "," ler "." ile değiştirildi.

data["coastline_coastarea_ratio"]=data["coastline_coastarea_ratio"].str.replace(',','.')
data["pop_density_per_sq_mi"]=data["pop_density_per_sq_mi"].str.replace(',','.')
data["net_migration"]=data["net_migration"].str.replace(',','.')
data["infant_mortality_per_1000_births"]=data["infant_mortality_per_1000_births"].str.replace(',','.')
data["literacy_"]=data["literacy_"].str.replace(',','.')
data["phones_per_1000"]=data["phones_per_1000"].str.replace(',','.')
data["arable_"]=data["arable_"].str.replace(',','.')
data["crops_"]=data["crops_"].str.replace(',','.')
data["other_"]=data["other_"].str.replace(',','.')
data["climate"]=data["climate"].str.replace(',','.')
data["birthrate"]=data["birthrate"].str.replace(',','.')
data["deathrate"]=data["deathrate"].str.replace(',','.')
data["agriculture"]=data["agriculture"].str.replace(',','.')
data["industry"]=data["industry"].str.replace(',','.')
data["service"]=data["service"].str.replace(',','.')
#ust sectionda temizlenen verilerde tip dönüşümü uygulandı.

data["coastline_coastarea_ratio"]=data["coastline_coastarea_ratio"].astype(float)
data["pop_density_per_sq_mi"]=data["pop_density_per_sq_mi"].astype(float)
data["net_migration"]=data["net_migration"].astype(float)
data["infant_mortality_per_1000_births"]=data["infant_mortality_per_1000_births"].astype(float)
data["literacy_"]=data["literacy_"].astype(float)
data["phones_per_1000"]=data["phones_per_1000"].astype(float)
data["arable_"]=data["arable_"].astype(float)
data["crops_"]=data["crops_"].astype(float)
data["other_"]=data["other_"].astype(float)
data["climate"]=data["climate"].astype(float)
data["birthrate"]=data["birthrate"].astype(float)
data["deathrate"]=data["deathrate"].astype(float)
data["agriculture"]=data["agriculture"].astype(float)
data["industry"]=data["industry"].astype(float)
data["service"]=data["service"].astype(float)
def eksik_degerler(data): 
    bos_deger_toplami = data.isnull().sum()
    bos_deger_yuzdesi= 100 * data.isnull().sum()/len(data)
    bos_deger_tablosu = pd.concat([bos_deger_toplami, bos_deger_yuzdesi], axis=1)
    bos_deger_tab_yeni_isim = bos_deger_tablosu.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return bos_deger_tab_yeni_isim
eksik_degerler(data)
#other adlı satırın işimize yaramayacağını düşündük ve uçurduk.
data.drop('other_',axis=1, inplace=True)
# climate isimli featuredaki bilgiler 1/1,5/2/2,5/3/4 şeklinde olduğu için ve 
# bu değerlerin ortalaması yaklaşık 2 olduğu için ve bu değerler sınıflandırıcı değerler olduğundan
# NaN değerler 2 ile dolduruldu
data['climate'] = data['climate'].fillna(2)
#data.climate.value_counts()
#data.climate.unique()
#asagıdaki degerlerin NaN oranları yuksek oldugu icin ortalama ile dolduruldu.
#cok yuksek oranları yok fakat bizim verisetimiz icin en yuksek olanları sectik.

# agriculture degeri ortalama ile dolduruldu
data['agriculture'] = data['agriculture'].fillna(data.agriculture.mean())
# industry degerleri ortalama ile dolduruldu
data.industry=data.industry.fillna(data.industry.mean())
# servis degerleri ortalama ile dolduruldu
data.service= data.service.fillna(data.service.mean())
# net migration degerleri ortalama ile dolduruldu
data.literacy_=data.literacy_.fillna(data.literacy_.mean())
#diger featureların NaN degerleri cok az olduğu için onları 0 ile doldurduk
data.fillna(data.mean(),inplace=True)
# ülkelerin okur yazarlık oranına göre o ülke gelişime 1=Gelişime açık/0=Gelişime açık değil şeklinde yeni feature yaratıldı.
limit=data.literacy_.mean()
data["evaluation"]=["1" if i>limit else "0" for i in data.literacy_]
#limit
data.describe()  #data ile ilgili istatistiksel verileri görüntüler.
                 #toplam veri sayısı,ortalama,standart sapma,minumum değer,maximum değer,
                 #1.Çeyrek,Medyan ve 3.Çeyrek bilgilerini gösterir.
data.shape   #datanın kaç satır ve sütundan oluştuğunu gösterir. 
data.columns #data sütunlarını görüntülemek için kullanırız.
data.dtypes #featureların data tiplerini görüntüler.
data.corr()
#Yukarıda datamıza ait outleir değerleri görmek için bir boxplot çizdirdik.
#Bu boxplot ta 2 numaralı sınıfta çok yoğun olarak outlier olduğunu görüyoruz.
#Bu outlier değerler bizim datamızın doğruluğunu bozabilir.
#Eğer daha iyi bir sonuç elde etmek istiyorsak outlier değerlerimizi daha düşük seviyeye çekecek şekilde ayarlamamız gerekir.
# Seaborn Correlation Heatmap
f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(data.corr() , annot=True ,linewidths=.5,fmt=".2f")
plt.title("Countries of the World")
plt.show()
#BoxPlot
sns.boxplot(x=data['gdp_$_per_capita'])
#BoxPlot
print(data.boxplot(column="gdp_$_per_capita",by="climate"))
#WordCloud
# Hangi ülkelerden verilerden yoğun bulunduğunu görüntülemek isteyebiliriz.
plt.subplots(figsize=(10,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(data["country"]))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
#Min-Max Normalization
#Formule : x-min(x) / max(x)-min(x)
data["new_gdp_per_capita"]=(data["gdp_$_per_capita"]-min(data["gdp_$_per_capita"]))/((max(data["gdp_$_per_capita"]))-(min(data["gdp_$_per_capita"])))
data["new_phones_per_1000"]=((data["phones_per_1000"])-(min(data["phones_per_1000"])))/((max(data["phones_per_1000"]))-(min(data["phones_per_1000"])))
#Line Plot
data["new_gdp_per_capita"].plot(kind="line",color="blue",linestyle=":",label="GDP $ Per Capita",grid=True,alpha=0.8,figsize=(20,8))
data["new_phones_per_1000"].plot(kind="line",color="red",linestyle=":",label="Phones Per 1000",grid=True,alpha=0.8,figsize=(20,8))
plt.legend()
plt.title("DATA OF GDP PER CAPITA AND PHONES PER 1000",size=15,color="blue")
plt.show()
#Scatter Plot
data.plot(kind="scatter",x="new_gdp_per_capita",y="new_phones_per_1000",color="red",grid=True,linestyle="-",figsize=(20,8))
plt.title("GDP PER CAPITA AND PHONES PER 1000",size=15,color="red")
plt.xlabel("GDP PER CAPITA",color="red",size=12)
plt.ylabel("PHONES PER 1000",color="red",size=12)
plt.show()
#Histogram
data["birthrate"].plot(kind="hist",color="orange",bins=30,grid=True,alpha=0.4,label="Birthrate",figsize=(20,8))
plt.legend()
plt.xlabel("Birthrate",color="brown",size=12)
plt.ylabel("Frequency",color="brown",size=12)
plt.title("Birthrate Distribution")
plt.show()
data.head()
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

# Eğitim için ilgili öznitlelik değerlerini seç
X = data.iloc[:,6:11].values

# Sınıflandırma öznitelik değerlerini seç
Y = data.evaluation.values
# Eğitim ve doğrulama veri kümelerinin ayrıştırılması
validation_size = 0.20
seed = 7
X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)
# Naive Bayes modelini oluşturuyoruz.
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
# NB modelinin K-katlamalı çapraz doğrulama ile ACC değerinin hesaplanması
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
cv_results
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
msg
# Model ile ilgili bilgileri görüntülüyoruz.
# Precision , Recall , F1-score , Karmaşıklık matrisi değerlerini görüyoruz.
from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))
# Eğitim için X ve Y değerlerini aldık.
Y = data["new_phones_per_1000"].values
Y=Y.reshape(-1,1)
X = data["new_gdp_per_capita"].values
X=X.reshape(-1,1)
#X
#Y
# Test değerlerini,oranını belirliyoruz yani datamızı ayırıyoruz.
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)  
# Eğitim işlemini gerçekleştiriyoruz.
from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train, y_train)  
# Modelimiz hakkındaki bazı bilgiler.
print("Kesim noktası:", model.intercept_)  
print("Eğim:", model.coef_)
print(X_test)
y_pred = model.predict(X_test) 
# Modelimize bütün değerlerinin en yakınından geçen line'ı çizdiriyoruz.
plt.scatter(X_train, y_train, color = 'red')
modelin_tahmin_ettigi_y = model.predict(X_train)
#plt.scatter(X_train, modelin_tahmin_ettigi_y, color = 'blue')
plt.plot(X_train, modelin_tahmin_ettigi_y, color = 'aqua')
plt.title('Phones Per 1000 - Gdp Per Capita')
plt.xlabel("Gdp Per Capita")
plt.ylabel("Phones Per 1000")
plt.show()
# Eğitiğimiz modelin sonuçlarını görüyoruz.
# Bunun için En küçük kareler yöntemi ve Kök ortalama kare hatası yöntemlerini kullanıyoruz.
# Burada modelimiz baya iyi sonuç vermiş gibi duruyor çünkü modelimizin MSE değeri çok düşük.
from sklearn import metrics   
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))