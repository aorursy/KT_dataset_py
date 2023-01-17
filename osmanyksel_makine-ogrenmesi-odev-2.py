
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

#Dosya okuma 
file='../input/NBA_player_of_the_week.csv'
data=pd.read_csv(file)
#Basit  istatistikler
data.describe()
#Bellek kullanımı ve veri türleri
data.info()
#İlk beş satır
data.head()
#Son beş satır
data.tail()
#Rastgele beş satır
data.sample(5)
#Satır ve sütun sayısı
data.shape
#Histogram bakma
data.hist()
#Korelasyon Matrisi
data.corr()
#Korelasyon Gösterim seaborn ısı haritası ile
import seaborn as sns
corr=data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#Korelasyonları yüksek iki öznitelik için plotting işlemi-(Age,Seasons in league)
import matplotlib.pyplot as plt
plt.plot(data['Age'],data['Seasons in league'])
plt.xlabel('Age')
plt.ylabel('Seasons in league')
plt.show()

#Eksik değerleri buluyoruz
data.isnull().sum()
#Null olan özniteliklere sahip, toplam kayıt sayısını buluyoruz
data.isnull().sum().sum()
#Eksik değer tablosu
def eksikDegerTablosu(data):
    mis_val=data.isnull().sum()
    mis_val_percent=100*data.isnull().sum()/len(data)
    mis_val_table=pd.concat([mis_val,mis_val_percent],axis=1)
    mis_val_table_ren_columns=mis_val_table.rename(columns={0:'Eksik Değerler',1:'Değeri % cinsinden'})
    return mis_val_table_ren_columns
#Eksik değer tablosunu çağırıyoruz
eksikDegerTablosu(data)
data
#%70 üzerinde null olan kolonları sil
tr=len(data)*.3
data.dropna(thresh=tr,axis=1,inplace=True)
data
#Conference deki null degerleri belirtilmedi olarak doldur
data['Conference']=data['Conference'].fillna('Belirtilmedi')
data
#Aykırı uç değer tespiti
import seaborn as sns 
sns.boxplot(x=data['Age'])

p=np.percentile(data.Age,[1,99])
p
new_data=data[(data.Age>p[0]) & (data.Age<p[1])]
new_data
data
#Mevcut öznitelikten yeni bir öznitelik yaratma
import datetime
dir(datetime.datetime)

an=datetime.datetime.now()
yil=an.year

data['Dogum_Yili']=yil-data['Age']
data
#Veri normalleştirme
from sklearn import preprocessing
#Age özniteliğini normalleştirmek istiyoruz
x=data[['Age']].values.astype(float)
#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler=preprocessing.MinMaxScaler()
x_scaled=min_max_scaler.fit_transform(x)
data['Age2'] = pd.DataFrame(x_scaled)

data
#Uygun eğitim ve sınıflandırma öznitelik seçimi
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

#Eğitim için ilgili öznitlelik değerlerini seç
X = data.iloc[:,[0,3]].values

#Sınıflandırma öznitelik değerlerini seç
Y = data.iloc[:, -2].values

Y
X
#Naive Bayes modelinin oluşturulması
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


#Naive bayes-Sonuçlar
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier #Sınıflandırıcı tarafından yapılan tahminlerin özeti
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))
#Linear regresyon uygulanması ve sonuçları
veri={'Age':data['Age'],
      'Seasons in league':data['Seasons in league']}

df = pd.DataFrame(veri)
df
df.plot(x='Age', y='Seasons in league', style='o')  
plt.title('Age - Seasons in league')  
plt.xlabel('Age')  
plt.ylabel('Seasons in league')  
plt.show() 
Y = df.iloc[:, 0].values  
X = df.iloc[:, 1:].values   
Y
X
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 
from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train, y_train) 
print("Kesim noktası:", model.intercept_)  
print("Eğim:", model.coef_)
X_test
y_pred = model.predict(X_test) 
df = pd.DataFrame({'Gerçek': y_test, 'Tahmin Edilen': y_pred})  
df 
plt.scatter(X_train, y_train, color = 'red')
modelin_tahmin_ettigi_y = model.predict(X_train)
#plt.scatter(X_train, modelin_tahmin_ettigi_y, color = 'blue')
plt.plot(X_train, modelin_tahmin_ettigi_y, color = 'blue')
plt.title('Age - Seasons in league')
plt.xlabel('Age')
plt.ylabel('Seasons in league')
plt.show()
#Linear regresyon-Sonuçlar
from sklearn import metrics   
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))