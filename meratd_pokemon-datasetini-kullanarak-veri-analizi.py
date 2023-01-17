# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import random 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
random.seed(1)
#Csv dosyasından veri okuma
data=pd.read_csv("../input/pokemon.csv")
combat = pd.read_csv('../input/combats.csv')
tests = pd.read_csv('../input/tests.csv')
#ilk 5 satır
data.head()
# '#' işaretini 'Number' kelimesiyle değiştirilmesi.
data = data.rename(index=str, columns={"#": "Number"})
#ilk 5 satır
data.head()
#bellek kullanımı ve veri türleri
data.info()
#son 5 satır
data.tail()
#basit istatistikler
data.describe()
#satır ve sütun sayısı
data.shape
#Histogram grafiği
data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# Korelasyon haritası
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Çizgi Grafiği
data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = etiketi grafiğe koyar
plt.xlabel('x ekseni')              # label = etiket adı
plt.ylabel('y ekseni')
plt.title('Çizgi Grafiği')            # title = grafiğin başlığı
plt.show()
#Null olan özniteliklere sahip, toplam kayıt sayısını buluyoruz
data.isnull().sum()
#Name sütunundaki boş yerlere 'Bos' kelime yazdırıyoruz.
data['Name'] = data['Name'].fillna('Bos')
data.head()
#'Type 2' deki bos değerleri 'Type 1' değerleriyle dolduruyoruz.
data['Type 2'].fillna(data['Type 1'],inplace=True)
data.info()
#Legendary sütunundaki true false değerlerine 1 ve 0 atıyoruz
data['Legendary'] = data['Legendary'].map({False: 0, True:1})
data.head()
#2.Aykırı Değer Tespiti
sns.boxplot(x=data['HP']) 
sns.boxplot(x=data['Attack'])
#Mevcut özniteliklerden yeni bir öznitelik oluşturma
def guc_durumu(Attack):
    return (Attack >= 60)

data['Power'] = data['Attack'].apply(guc_durumu)
data
#Veri Normalleştirme
from sklearn import preprocessing

#Puan özniteliğini normalleştirmek istiyoruz
x = data[['Attack']].values.astype(float)

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data['Attack2'] = pd.DataFrame(x_scaled)
#datamızdaki bütün veriler
data
data.info()
#datadaki veriler veri tipleri
data.dtypes
data.shape
#MODELLEŞTİRME
data.dropna(axis=0, how='any')
# Veri kümesini Eğitim seti ve Test kümesine ayırdık
X = data.iloc[:, 5:11].values
y = data.iloc[:, 11].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Kullanacağımız modeller için kullanacağımız kütüphaneler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
#Modellerin eğitilmesi
models = []
models.append(('Naive Bayes', GaussianNB()))
models.append(('Logistic Regression', LogisticRegression()))
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFE

# Modelleri test edelim
for name, model in models:
    model = model.fit(X_train, y_train)
    Y_pred = model.predict(X_test)
    
    #Accuracy değeri gör
    print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(y_test, Y_pred)*100))
    
    #Confusion matris görmek için aşağıdaki kod satırlarını kullanabilirsiniz   
    report = classification_report(y_test, Y_pred)
    print(report)
    