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
df=pd.read_csv("../input/student-mat.csv")
df.head()# ilk 5 satır
df.tail() #tablodaki son 5 satır
df.describe() #basit istatistikler ve cevapları
df.shape #satir ve sutun sayisi
df.info()
df.isnull().sum() #boş deger kontrolü
#Aykırı Değer Tespiti
import seaborn as sns
sns.boxplot(x=df['G1'])
#first period grade (numeric: from 0 to 20)
#Histogram grafiği incelemesi
df.hist()
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
df.cov() # covaryans baktık 
df.corr() # corelasyon bakıyoruz değerleri değerlendircek olursak G1- G3 arasında pozitif yönlü kuvvetli bir ilişki var 
df.plot(x='G2', y='G3', style='*')  # G2 G3 arasındaki ilişkiyi görebilmek için plotting çizimi yapıyoruz düzenli bi dağılım söz konusu biri artıyorsa diğeride artıyor
from datetime import datetime   # Mevcut 'age' özniteliğinden yeni bir 'dogumYili' özniteliği oluşturma
year=pd.datetime.now()
dogumYili=year.year-df['age']
df['dogumYili']=dogumYili
df
 # veri normalleştirme
from sklearn import preprocessing

#G1 özniteliğini normalleştirmek istiyoruz
x = df[['G1']].values.astype(float)

#Normalleştirme için MinMax normalleştirme yöntemini kullanıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['G1_new'] = pd.DataFrame(x_scaled)
df
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection
#Eğitim için ilgili öznitlelik değerlerini seç
X = df.iloc[:,:-1].values
Y = df.iloc[:, ].values
X
X[:,3]
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 1]   #cinsiyet değerlerini char biçiminden çıkartıp algoritmanın anlayabilceği 0 ve 1 lere dönüştürdük- 0=F / 1=M
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X[:, 3]   #student's home address type (binary: 'U' - urban or 'R' - rural)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 4] = labelencoder.fit_transform(X[:, 4])
X[:, 4] #family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3) Burda aile boyutu 3 ten az oldugu durumlarda 1 çok odugu durumlarda 0
X[:, 5]
# Encoding categorical data 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 5] = labelencoder.fit_transform(X[:, 5])
X[:, 5]  # ebeveynlerin birlikte yaşama durumu Ayrı olduklarında 0 birlikte olduklarında 1
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import model_selection
#Eğitim için ilgili öznitlelik değerlerini seç
X = df.iloc[:, 30:33] 
#Sınıflandırma öznitelik değerlerini seç
Y = df.iloc[:, 30]
X
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
cv_results
msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
msg
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print("ACC: ",accuracy_score(Y_pred,Y_test))
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)  
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df2
df3 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df3
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) # ortlama kök kare hatası ve ortalama kare hatası sonuçları



