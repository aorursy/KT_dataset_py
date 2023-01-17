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
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import os as o


columns= [
    'Apps',
    'Category',
    'Rating',
    'Reviews',
    'Size',
    'Installs',
    'Type',
    'Price',
    'Content Rating',
    'Genres',
    'Last Updated',
    'Current Ver',
    'Android Ver'
]
df=pd.read_csv('../input/googleplaystore.csv')

kfold = KFold(3, True, 1)

#VERI KESFİ VE GORSELLESTIRME....
df.head()
df.dtypes
df.isnull().any()
df.shape
df.describe()
#Veri ön işleme adım 1
mapping = {'no': 0., 'yes':1., 'False.':0., 'True.':1.}
df.replace({'international plan' : mapping, 'voice mail plan' : mapping, 'googleplaystore' : mapping}, regex = True, inplace = True)
#Veri ön işleme adım 2 
#df.drop('', axis = 1, inplace = True)
#kaldirilacak öznitelik olmadıgı icin drop islemini gerceklestirmiyoruz.
df.shape
df.describe()
df.info()
df.tail(6) #Son 6 satırı getirme
df.sample(10) #rassal 10 satır getirme
#Histogram grafiği görmek için 
import matplotlib.pyplot as plot
num_bins = 20
df.hist(bins=num_bins, figsize=(5,5))
#Korelasyon Gösterim 1
import matplotlib.pyplot as plot
plot.matshow(df.corr())

#Korelasyonuzmuzda tek bağımsız değişken olan ratingin’in kullanılması bize tek renk bir grafik çıkartmış bulunmakta.
#Renginin koyu olması ise + yönünde pozitif bir artan korelasyon varlığından söz ediyor.
#Korelasyon Gösterim 2
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#ON ISLEME...
df.isnull().sum()
#Null olan özniteliklere sahip, toplam kayıt sayısını buluyoruz
df.isnull().sum().sum()
#Eksik değer tablosu
def eksik_deger_tablosu(df): 
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns
eksik_deger_tablosu(df)
#Null olan öznitelikleri buluyoruz ve üzerlerinde eksik veri doldurma işlemleri gerçekleştireceğiz.
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X[:, 3] 
#Groupby kullanımı
df.groupby('App')['Category'].apply(lambda x: x.count())
#Groupby kullanımı2
df.groupby('App')['Rating'].apply(lambda x: np.mean(x))
#İstatiksel fonksiyonlar kullanımı:

df['Rating'].mean()
#DataFrame'de bulunan sayısal verilerin ortalamasi:
df.mean(axis=0,skipna=True)

#Rating features'inin medyanının bulunması:
df['Rating'].median()
#Rating Features'ının modunun bulunması:
df['Rating'].mode()
#Rating Fetarue'sının standart sapmasının bulunması:
df['Rating'].std()
#Kovaryans Matrisi Hazırlıyoruz....
df.cov()
df.corr()
df.plot(x='Type', y='Rating', style='o')
#Eksik değer tablosu
def eksik_deger_tablosu(df): 
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns
df
#%70 üzerinde null değer içeren kolonları siliyoruz ve alt satırda sildiğimizi gösteriyoruz
tr = len(df) * .3
df.dropna(thresh = tr, axis = 1, inplace = True)
df
#Rating kolonundaki Null değerleri 'boş' değeri ile dolduruluyor...
df['Rating'] = df['Rating'].fillna('boş')
df
#Andorid Ver kolonundaki Null değerleri 'boş' değeri ile dolduruluyor...
df['Android Ver'] = df['Android Ver'].fillna('boş')
df
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X[:, 3] 
df1=df[df['Type']=='Free']
df2=df[df['Type']=='Paid']
#Ozniteliklerde Type'ı Free olanlar:
df1.shape
#Ozniteliklerde Type'ı Paid olanlar:
df2.shape
df = df2.append(df1[:800])
#def populerlikDurumu(indirme):
    #return (indirme >= 70)

#df['Populer'] = df['indirme'].apply(populerlikDurumu)

#df['populer'] = df['Installs']
#df


#import datetime as d
#populerlikDurumu = (df['Installs']+ df['Rating'])
#df['Popularity'] = populerlikDurumu 
#df

df['popularity']= df['Installs']
df
df.shape
df.corr()
import pandas
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

#Eğitim için ilgili öznitlelik değerlerini seçilmesi
X = df.iloc[:, :-1].values

#Sınıflandırma öznitelik değerlerini seçilmesi
Y = df.iloc[:, -1].values
X
Y
#MODEL EGİTİMİ.....
#Eğitim ve doğrulama veri kümelerinin ayrıştırılması
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
X_train
X_validation
Y_validation
Y_train
X
Y
#3.Veri Normalleştirme
from sklearn import preprocessing

#Puan özniteliğini normalleştirmek istiyoruz
x = df[['Rating']].values.astype(str)

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['Rating2'] = pd.DataFrame(x_scaled)

df
#Naive Bayes modelinin oluşturulması
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
#NB modelinin K-katlamalı çapraz doğrulama ile ACC değerinin hesaplanması
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)


print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))

from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(Y_pred,Y_test))
cv_results
from sklearn.metrics import r2_score
import pandas as pd
#Categorylere göre App'leri grupluyoruz...
New_df = pd.DataFrame(df.groupby("Category")["App"].sum())
#App'leri categorylerine göre sınıflandırıyoruz.
New_df.head()
New_df.tail()
#Regresyon Modeli Hazırlık Aşaması:
#x = pd.DataFrame(new_df.index)
#y = New_df.iloc[:, 0]
#x = x.iloc[:-1, :].values.reshape(-1,1)
#y = New_df.iloc[:-1, 0].values.reshape(-1,1)

#Model oluşturmamız için gereken kütüphaneler:
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#lineer Regresyon için modelimizi oluşturuyoruz.
lr = LinearRegression()

# Derecesi 4 olan bir fonksiyon kullanacağız
pf = PolynomialFeatures(degree=4)
x_pol = pf.fit_transform(x)