#Gerekli Kütüphaneler Geliştirme Ortamına Dahil Ediliyor

import numpy as np # linear algebra

import pandas as pd # Veri işleme



# Visiualization tools

import matplotlib.pyplot as plt

import seaborn as sns



#Model Seçimi

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score





#Makine Öğrenmesi Modelleri

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC









#Metrikler

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report



from sklearn.externals import joblib



#Sistem Kütüphaneleri

import os

print(os.listdir("../input"))
import warnings

#Sonuçların okunmasını zorlaştırdığı için uyarıları kapatıyoruz

warnings.filterwarnings("ignore")

print("Uyarılar Kapatıldı")
#veri setini pandas DataFrame olarak yüklüyoruz

dataset=pd.read_csv('../input/indian_liver_patient.csv')



#veri setine ait ilk beş satır; 

dataset.head()
# veri setindeki sayısal özelliklere ait özet istatistiksel bilgiler

# Gender özelliği sayısal olmayan değerler içerdiği için,istatistiksel verileri yoktur

dataset.describe().T
dataset.info()
#eksik veriler 'Albumin_and_Globulin_Ratio' sütununun ortalaması ile doldurulu

dataset['Albumin_and_Globulin_Ratio'].fillna(dataset['Albumin_and_Globulin_Ratio'].mean(), inplace=True)

dataset.info()
#'Dataset' sütünun adı 'target' olarak değitiriliyor

dataset.rename(columns={'Dataset':'target'},inplace=True)

dataset.head()
target_counts=dataset['target'].value_counts().values

gender_counts=dataset['Gender'].value_counts().values



fig1, axes=plt.subplots(nrows=1, ncols=2,figsize=(10,5))

fig1.suptitle("Teşhis ve Cinsiyet Yüzdeleri")



target_sizes=dataset.groupby('target').size()

axes[0].pie(

    x=target_counts,

    labels=['patient({})'.format(target_sizes[1]),'not patient({})'.format(target_sizes[2])],

    autopct='%1.1f%%'

)

axes[0].set_title("Hasta Teşhis Yüzdeleri")



gender_sizes=dataset.groupby('Gender').size()

axes[1].pie(

    x=gender_counts, 

    labels=['male({})'.format(gender_sizes['Male']), 'female({})'.format(gender_sizes['Female'])], 

    autopct="%1.1f%%"

)

axes[1].set_title("Hastaların Cinsiyet Yüzdeleri")
dataset=pd.get_dummies(dataset)
corr_matrix=dataset.corr()

fig, ax = plt.subplots(figsize=(12,12))

sns.heatmap(corr_matrix,annot=True,linewidths=.5, ax=ax)
#Veri seti data ve target olarak ayrıştırılır

X=dataset.drop('target', axis=1) #data

y=dataset['target'] # target
#Eğitim ve test kümelerine ayrıştırılır

X_train, X_test, y_train, y_test=train_test_split(X,y, stratify=y, test_size=0.3,random_state=42)
models=[]

models.append(("Logistic Regression",LogisticRegression()))

models.append(("Naive Bayes",GaussianNB()))

models.append(("KNN",KNeighborsClassifier(n_neighbors=5)))

models.append(("Decision Tree",DecisionTreeClassifier()))

models.append(("SVM",SVC()))



for name, model in models:

    

    clf=model



    clf.fit(X_train, y_train)



    y_pred=clf.predict(X_test)

    print(10*"=","{} için Sonuçlar".format(name).upper(),10*"=")

    print("Accuracy:{:0.2f}".format(accuracy_score(y_test, y_pred)))

    print("Confusion Matrix:\n{}".format(confusion_matrix(y_test, y_pred)))

    print("Classification Report:\n{}".format(classification_report(y_test,y_pred)))

    print(30*"=")