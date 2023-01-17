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
df = pd.read_csv('../input/winequality-red.csv')
df
df.describe() #basit istatistikler
df.head() #ilk 5 satır
df.tail() #son 5 satır  
df.sample(5) #rassal 5 satır
df.shape #satır ve sütun sayısı
df.info() #bellek kullanımı ve veri türleri
#Sayısal tüm özniteliklerin ortalaması
df.mean(axis=0,skipna=True)
df.hist()
df.plot(x='alcohol', y='quality', style='.')  
plt.title('alcohol - quality')  
plt.xlabel('alcohol')  
plt.ylabel('quality')  
plt.show() 
#Korelasyon Gösterim
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#Eksik değer tablosu
def eksik_deger_tablosu(df): 
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Eksik Değerler', 1 : '% Değeri'})
    return mis_val_table_ren_columns
eksik_deger_tablosu(df)
#elde edilen veriler türetilmiş ve düzenlenmiş verilerdir. Eksik veya hatalı veri bulunmamaktadır...


#%70 üzerinde null değer içeren kolonları sil
#tr = len(df) * .3
#df.dropna(thresh = tr, axis = 1, inplace = True)
#pH kolonundaki Null değerleri 'NOTR' değeri ile doldur
#df['pH'] = df['pH'].fillna('NOTR')

#2.Aykırı Değer Tespiti
import seaborn as sns
sns.boxplot(x=df['quality'])
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection

#Eğitim için ilgili öznitlelik değerlerini seç
X = df.iloc[:, :-1].values

#Sınıflandırma öznitelik değerlerini seç
Y = df.iloc[:, -1].values

Y
#Eğitim ve doğrulama veri kümelerinin ayrıştırılması
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


#Naive Bayes modelinin oluşturulması
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

#NB modelinin K-katlamalı çapraz doğrulama ile ACC değerinin hesaplanması
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)


cv_results #acc değerleri
msg = "%f (%f)" % (cv_results.mean(), cv_results.std()) 
msg
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(y_pred,y_test))