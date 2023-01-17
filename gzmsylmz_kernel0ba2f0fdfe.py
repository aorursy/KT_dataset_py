# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#1-VERİ KEŞFİ VE GÖRSELLEŞTİRME
import numpy as np
import pandas as pd
data = pd.read_csv("../input/winequality-red.csv")
data.describe() #verilerin basit istatistikleri

data.info() #bellek kullanımı ve veri türlerini veriyor
data.head() #Tablonun ilk 5 deki bilgilerini verir
data.tail() #Tablodaki son 5 deki bilgilerini verir
data.shape #Kaç satır ve sütundan oluştuğunu gösterir
data.hist()#Histogram grafiği
import matplotlib.pyplot as plt
num_bins = 10
data.hist(bins=num_bins, figsize=(20,15)) #Başka bir Histogram grafiği
import seaborn as sns
corr=data.corr()
data.corr() #düz metin halinde gösterme
#seaborn ısı haritası
import seaborn as sns
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#Kutu çizim grafiği incelemesi
#data.plot(kind='box', sharex=False, sharey=False)

data.plot(kind='box', subplots=True, layout=(7,7), sharex=False, sharey=False) 
data.plot(kind='scatter', x='pH', y='alcohol',alpha = 0.5,color = 'red', figsize=(9,9))
plt.xlabel('pH')             
plt.ylabel('alcohol')
plt.title('pH & alcohol')        
plt.show()
#Verilerin genellikle yaklaşık olduğunu görüyoruz bu bizim grafiğimizin iyi olduğunu gösteriyor.
#Eğer noktalar çok ayrık olsaydı o zaman grafiğimiz düzgün olmazdı.
data.pH.plot(kind = 'hist',bins = 100,figsize = (9,9))
plt.show() #Bu da farklı bir grafiğimiz
data.pH.plot(kind='line', color='g', label='pH', linewidth=2,alpha=0.5, grid=True,linestyle=':')
data["alcohol"].plot(color='r',label='quality',linewidth=5, alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('alcohol')
plt.ylabel('pH')
plt.title('Line Plot')
plt.show()
#Bu grafikte de alcohol verileriyle pH ve alcohol verileriyle quality karşılaştırıyoruz.
#Aralaraındaki değerlere göre böyle bir grafik çıkıyor.
#Grafikten anlaşılacağı gibi ph ve alcohol verileri daha yakın olduğu için noktalar daha yakın çıkmış.
#2-ÖN İŞLEME
#Eksik Değer Kontrolü
#null olan değerleri buluyor
data.isnull().sum()
# Bizim veri setimizde null oland değer yokmuş onun için null olan değerlerin içerisini 
#doldurmamıza gerek yok.
#null olan değerlerin toplamı
data.isnull().sum().sum()
#Eksik Değer Tablosu
def eksik_deger_tablosu(data): 
    mis_val = data.isnull().sum()
    mis_val_percent = 100 * data.isnull().sum()/len(data)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns
eksik_deger_tablosu(data)
#Bizim eksik değerimizin olmadığını tablodan görebiliyoruz ve % değerimizde de null olmadığı için 
# 0 oluyor.
#Uç Değerleri Bulma
#max
data.max()
#min
data.min()
#Burada veriler arasından en büyük (max) uç değeri ve veriler arasından en küçük
#(min) uç değerlerimizi buluyoruz.
#Mevcut Öznitelik yeni Öznitelik Oluşturma
import datetime as d
digerAsit= (data['pH']+data['density'])
data['DigerAsitler']=digerAsit
data
#Burada da yeni özniteliğimizin adı DigerAsitler 
#Öznitelikten bir tanesini normalleştirme
from sklearn import preprocessing

#Uzunluk özniteliğini normalleştirmek istiyoruz
x = data[['DigerAsitler']].values.astype(float)

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data['Veriler'] = pd.DataFrame(x_scaled)

data
#3-MODEL EĞİTİMİ
#1-Linear Regression Modeli
import matplotlib.pyplot as plt 
data.describe()
data.plot(x='pH', y='free sulfur dioxide', style='o')  
plt.title('pH - free sulfur dioxide')  
plt.xlabel('pH')  
plt.ylabel('free sulfur dioxide')  
plt.show() 
#burda grafiklerin doğru çıkmadığını görüyoruz
Y = data.iloc[:,8].values.reshape(-1,1)  #y_kolonunun çekilmesi->pH değeri
pd.DataFrame(Y).shape
x_data=data.drop(['pH'],axis=1)

X = x_data.iloc[:,0:11].values  #X değerlernin alınması
pd.DataFrame(X).shape
scale_oncesiX=X
scale_oncesiY=Y
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X=sc_X.fit_transform(X)
Y=sc_Y.fit_transform(Y)
#x ve y değerler,nin belli bir aralığa indirilmesi
X
Y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
pd.DataFrame(X_train).shape
pd.DataFrame(y_train).shape
pd.DataFrame(X_test).shape
pd.DataFrame(y_test).shape
from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train, y_train)  #modelin eğitilmesi
print("Kesim noktası:", model.intercept_) 
print("Eğim:", model.coef_)
modelin_tahmin_ettigi_y = model.predict(X_test)#x_test değerlerine göre sonuçların tahmin edilmesi

y_pred=sc_Y.inverse_transform(modelin_tahmin_ettigi_y)#tahmin edilen değerler normalize halde olduğu için gerçek değerlere dönüştürme
y_pred
plt.scatter(X, Y, color = 'red')
#plt.scatter(X_train, modelin_tahmin_ettigi_y, color = 'blue')
plt.plot(X, sc_Y.inverse_transform(model.predict(X)), color = 'blue')
plt.title('pH - free sulfur dioxide')
plt.xlabel('free sulfur dioxide')
plt.ylabel('pH')
plt.show()
from sklearn import metrics   
print('Linear Regression Mean Squared Error (MSE):', metrics.mean_squared_error(sc_Y.inverse_transform(y_test), y_pred))  
print('Linear Regression Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(sc_Y.inverse_transform(y_test), y_pred)))  
#Lineer regresyon modeline göre mse ve rmse sonuçları
#2-SVR Modeli
from sklearn.svm import SVR
regressor= SVR(kernel="rbf")
regressor.fit(X_train,y_train)#SVR algoritması kullnılarak modelin eğitilmesi
y_pred2 = regressor.predict(X_test)#X_test değerlerine göre test edilmesi
y_pred2=sc_Y.inverse_transform(y_pred2)#normalize şekilde çıkan sonuçların gerçek haline dönüştürülmesi
print('Linear Regression Mean Squared Error (MSE):', metrics.mean_squared_error(sc_Y.inverse_transform(y_test), y_pred2))  
print('Linear Regression Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(sc_Y.inverse_transform(y_test), y_pred2)))
#SVR algoritmasına  göre mse ve rmse sonuçları
SvrLogKarsilastirma=pd.DataFrame(sc_Y.inverse_transform(y_test),columns=['Gerçek Değer'])
SvrLogKarsilastirma=SvrLogKarsilastirma.assign(LinearRegressionTahmin=y_pred)
SvrLogKarsilastirma=SvrLogKarsilastirma.assign(SVRTahmin=y_pred2)
print(pd.DataFrame(SvrLogKarsilastirma))
#Gerçek değer,Svr tahmin,Lineer Regresyon Tahmin karşılaştırması