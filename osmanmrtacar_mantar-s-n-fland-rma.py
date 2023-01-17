# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/mushrooms.csv')
data.head(5) #datasetteki ilk 5 veriyi yazdır.
data.tail(5) #datasetteki son 5 veriyi yazdir.
data.describe() #Özeti yazdır
data.shape #Kaç satır sütun var yazdır
data.info() #özet
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()   #Tablodaki bilgileri sayısal değere dönüştürüyoruz.
for col in data.columns:
    data[col] = le.fit_transform(data[col])
data.head()
data.hist()
corr=data.corr() #Korelasyon haritası
corr 
sns.heatmap(corr) #korelasyon ısı harıtası
sns.barplot(x="gill-size", y='class', hue="bruises", data=data)
data.isnull().sum() 
sns.boxplot(data['habitat']) 
from sklearn import preprocessing
X = data.drop('class', axis=1) #bize yardım edecek olan kolonlar
y = data['class'] #hedef
y.values
#Normalleştirme
x = data[['habitat']].values.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

data['habitatN'] = pd.DataFrame(x_scaled)
data.head()
#Elimizdeki (X,y) verisini %70 train %30 test şeklinde ayırıyoruz
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
#Modeli eğitiyoruz
LR.fit(X_train,y_train)
#Modelin tahmin yeteneğini test ediyoruz
predictions = LR.predict(X_test)
#Acc score
score = LR.score(X_test, y_test)
score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
#Karmaşıklık matrisi
cm = confusion_matrix(y_test, predictions)
print(cm)
#Precision recall vb.
classification_report(y_test,predictions)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
#Modeli eğitiyoruz
clf.fit(X_train,y_train)
#Modelin tahmin yetenegini test ediyoruz
clf_predict = clf.predict(X_test)
#ACC score
clf_score = clf.score(X_test, y_test)
clf_score
#Karmaşıklık matrisi
confusion_matrix(y_test, clf_predict)
#Precision recall vb.
classification_report(y_test,clf_predict)