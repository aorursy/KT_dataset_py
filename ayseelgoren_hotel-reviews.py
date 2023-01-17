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
import pandas as pd
import numpy as np
data = pd.read_csv("../input/hotel-reviews/Datafiniti_Hotel_Reviews.csv")
#Veri Keşfi ve Görselleştirme
#İlk 5 satırın getirilmesi. 
data.head()
#Bellek kullanımı ve veri türlerinin belirtilmesi.
data.info()
#Satır ve sütun sayısı
data.shape
#Basit istatistikler
data.describe()
#Son 5 satır
data.tail()
#Histogram bakma
data.hist()
#latitude , longitude ve reviews.rating değerlerinin aralıklarını belirtir.
#Korelasyonun seaborn ısı haritası ile gösterimi
import seaborn as sns
corr = data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#latitude değişkeni reviews.rating ile ilişki şiddeti kötü iken diğerleriyle daha iyi.
#longitude değişkeni reviews.rating ile ilişki şiddeti kötü iken diğerleriyle daha iyi.
#reviews.rating değişkeninin , longitude ve latitude ile ilişki şiddeti kötüdür.
#Korelasyonun düz metin ile gösterimi 
data.corr()
#Korelasyonları yüksek olan longitude ve latitude öznitelikleri için plotting (çizim işlemi) 
import matplotlib.pyplot as plt
data.plot(x='latitude', y='longitude', style='o')  
plt.title('Latitude - Longitude')  
plt.xlabel('latitude')  
plt.ylabel('longitude')  
plt.show() 
#Ön İşleme
#Eksik Değer Doldurma
#Null olan öznitelikleri buluyoruz
data.isnull().sum()
#Eksik değerlerin(reviews.userCity ve reviews.userProvince) doldurulması.
data['reviews.userCity'] = data['reviews.userCity'].fillna('boş')
data['reviews.userProvince'] = data['reviews.userProvince'].fillna('boş')
data.isnull().sum()
#Uç değerleri(reviews.rating) bulma 
import seaborn as sns
sns.boxplot(x=data['reviews.rating'])
#reviews.rating değişekeninin hangi uç değerlerleri aldığı belirtiliyor.
#Tarih alanındaki yıl bilgisini kullanarak 'reviews.hotelCheck-inDate' isimli yeni bir öznitelik oluşturuyoruz.
date = pd.to_datetime(data['reviews.date'])
data['reviews.hotelCheck-inDate'] = date.dt.year
data
#Öznitelik Normalleştirme
from sklearn import preprocessing

#reviews.rating(Değerlendirme Derecesi) özniteliğini normalleştirmek istiyoruz
x = data[['reviews.rating']].values.astype(float)

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data['reviews.rating_2'] = pd.DataFrame(x_scaled)
data
#Model Eğitimi
#Model eğitimi için WEATER datasından yararlanılmıştır.
import pandas as pd
import numpy as np

df = pd.read_csv("../input/weater/weather_data_nyc_centralpark_2016(1).csv")
df
#date değişkenini int olarak almadığı için date kolonunu siliyorum.
df = df.drop(['date'],axis=1)
df
#Model 1
#Model 1 de maksimum sıcaklık dikkate alınarak sınıflandırma yapılır ve veri kümesindeki değişkenler tanımlanır.
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

#Eğitim için öznitlelik değerleri seçilir
X = df.iloc[:, 1:].values

#Sınıflandırma için maximum temperature öznitelik değeri seçilir
Y = df.iloc[:,0].values
Y
X
#Naive Bayes modelinin oluşturulması
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Sınıflandırıcı tarafından yapılan tahminlerin özeti
# Karmaşıklık matrisi, ACC, Precision, Recall değerlerinin görülmesi
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Doğruluk puanı hesaplanır
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))
#En yüksek sıcaklık derecelerinin hangileri oldukları ve sıklıklarını belirtiyor.
#Model 2
#Model 2 de kar derinliği dikkate alınarak sınıflandırma yapılır ve veri kümesindeki değişkenler tanımlanır.
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

#Eğitim için öznitlelik değerleri seçilir
X = df.iloc[:, :-1].values

#Sınıflandırma için  snow depth öznitelik değeri seçilir
Y = df.iloc[:,-1].values
Y
X
#Naive Bayes modelinin oluşturulması
from sklearn.naive_bayes import GaussianNB
model = GaussianNB() 
#Test ve train verileri belirlenir.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Sınıflandırıcı tarafından yapılan tahminlerin özeti
# Karmaşıklık matrisi, ACC, Precision, Recall değerlerinin görülmesi
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Doğruluk puanı hesaplanır
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))
#Model 2 karmaşıklık matrisi, ACC, precision, recall değerlerinin Model 1'e göre daha yüksek olması durumundan kullanılması daha uygundur.