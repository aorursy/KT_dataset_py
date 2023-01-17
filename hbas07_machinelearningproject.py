# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/NBA_player_of_the_week.csv')
df
df.head()
df.describe()
df.tail()
df.shape
#Korelasyon görüntüleme

df.corr()

#Korelasyon matrisinde görüldüğü gibi Age-Seasons in league,Seasons short-Draft Year 
#öznitelikleri arasında aynı yönlü mükemmele yakın bir ilişki vardır.
#Draft Year-Real_value,Season short-Real_value öznitelikleri arasında ise zıt yönlü mükemmele yakın bir ilişki vardır.
#Season short-Age,Real_value-Age öznitelikleri arasındaki ise ilişki yok denebilecek kadar zayıftır

#Isı haritası görünümü
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#Korelasyonu yüksek olan öznitelik (Season in league - Age ) için plotting işlemi
import matplotlib.pyplot as plt
plt.scatter(df['Seasons in league'], df['Age'])
plt.xlabel('Seasons in league')  
plt.ylabel('Age') 
plt.show()

#Korelasyonu yüksek olan öznitelik (Season short - Draft Year ) için plotting işlemi
plt.scatter(df['Season short'], df['Draft Year'])
plt.xlabel('Season short')  
plt.ylabel('Draft Year') 
plt.show()
#Histogram grafiği
df.hist()
#VERİ ÖN İŞLEME
#Eksik veri var mı?
df.isnull().sum()
#Eksik değer doldurma aşama 1
#eksik değer tablosu oluştur
def edt(df): 
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns
#Eksik değer doldurma aşama 2
#eksik değer tablosunu görüntüle
edt(df) 

#Eksik değer tablosuna bakıldığında Conference özniteliğinde %33 gibi bir eksik verimiz var
#bu tarz durumlar makine öğreniminde pek istenmediğinden bu değerleri doldurmalıyız.
#Eksik değer doldurma aşama 3
#Conference kolonundaki boş verileri East ile doldur
df['Conference'] = df['Conference'].fillna('East')
df
#Age kolonunda uç değerler tespiti
import seaborn as sns
sns.boxplot(x=df['Age'])
#Age özniteliğini kullanarak Yaslı özniteliği eklemesi
def yas(Age):
    return (Age > 30 )

df['Yaslı'] = df['Age'].apply(yas)
df
# Normalleştirme işlemi
from sklearn import preprocessing

#Age özniteliğini normalleştirelim
x = df[['Age']].values.astype(int)

#MinMax normalleştirme kullanılaraka normalleştirme işlemi gerçekleştiriyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['New_Age_Value'] = pd.DataFrame(x_scaled)

df

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

#Eğitim için ilgili öznitelik değerlerini seç
X = df.iloc[:, 9:10].values

#Sınıflandırma öznitelik değerlerini seç
Y = df.iloc[:,0].values


#Eğitim ve doğrulama veri kümelerinin ayrıştırılması
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#Naive Bayes ve Logistic Regression modelinin oluşturulması
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
models = []
models.append(('Naive Bayes', GaussianNB()))
models.append(('Logistic Regression', LogisticRegression()))
model.fit(X,Y)
#Naive Bayes modelinin K-katlamalı çapraz doğrulama ile ACC değerinin hesaplanması
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=15, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
cv_results
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFE

# Modelleri test edelim
for name, model in models:
    model = model.fit(X_train, Y_train)
    Y_pred = model.predict(X_validation)
    
    #Accuracy değeri gör
    print("%s -> ACC: %%%.2f" % (name,metrics.accuracy_score(Y_validation, Y_pred)*100))
    
    #Confusion matris görmek için aşağıdaki kod satırlarını kullanabilirsiniz   
    report = classification_report(Y_validation, Y_pred)
    print(report)
    
    #ROC_Ciz(Y_test, Y_pred)
#age,height,seasons in league,weight özniteliklerini
#kullanarak bir model eğitmeyi planladık
#çabaladık,araştırdık ancak yapamadık.




