# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model.logistic import LogisticRegression

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import confusion_matrix, accuracy_score,f1_score

import pickle

from sklearn.externals import joblib
import pandas as pd
df=pd.read_csv("../input/bank-churn-modelling/Churn_Modelling.csv",index_col='RowNumber')

df = df.drop(["CustomerId","Surname"],axis=1)

df = df.replace({"Female":0,"Male":1,"France":0,"Germany":1,"Spain":2}) 
print(df.columns)

min_max_scaler = MinMaxScaler()



df = min_max_scaler.fit_transform(df)

df = pd.DataFrame(df)
egitimveri,validationveri = train_test_split(df,test_size=0.2,random_state=7)



egitimgirdi = egitimveri.drop(df.columns[10],axis=1)

egitimcikti = egitimveri[10]



valgirdi = validationveri.drop(df.columns[10],axis=1)

valcikti = validationveri[10]



chi2_selector = SelectKBest(chi2, k=5)

X_kbest = chi2_selector.fit_transform(egitimgirdi, egitimcikti)
print('Original number of features:', egitimgirdi.shape[1])

print('Reduced number of features:', X_kbest.shape[1])
models = []

models.append(("LR",LogisticRegression()))

models.append(("LDA",LinearDiscriminantAnalysis()))

models.append(("KNN",KNeighborsClassifier()))

models.append(("DCT",DecisionTreeClassifier()))

models.append(("GNB",GaussianNB()))

models.append(("SVC",SVC()))

models.append(("MLP",MLPClassifier()))

models.append(("ADB",AdaBoostClassifier()))

models.append(('RAF', RandomForestClassifier()))
a=KNeighborsClassifier(n_neighbors=2,p=3,metric="manhattan",weights="uniform").fit(egitimgirdi,egitimcikti)

#Eğitim setiyle model kurulması

a = a.fit(egitimgirdi,egitimcikti)

#Kurulan model'in test edilmesi

y_pred = a.predict(valgirdi)

#Çıkan doğruluk skoru ve Hata Matrisi



cm = confusion_matrix(valcikti, y_pred) 

print("KNN confusion_matrix:\n", cm)

print("KNN accuracy_score: ", accuracy_score(valcikti, y_pred)),

print("\nKNN f1_score:",f1_score(valcikti, y_pred)),

filename = 'traditionalml.sav'

pickle.dump(a, open(filename, 'wb'))

print("\n")
a=SVC(C=100,kernel="rbf",gamma=0.1).fit(egitimgirdi,egitimcikti)

#Eğitim setiyle model kurulması

a = a.fit(egitimgirdi,egitimcikti)

#Kurulan model'in test edilmesi

y_pred = a.predict(valgirdi)

#Çıkan doğruluk skoru ve Hata Matrisi



cm = confusion_matrix(valcikti, y_pred) 

print("SVC confusion_matrix:\n", cm)

print("SVC accuracy_score: ", accuracy_score(valcikti, y_pred)),

print("\nSVC f1_score:",f1_score(valcikti, y_pred)),

filename = 'traditionalml.sav'

pickle.dump(a, open(filename, 'wb'))

print("\n")
a=DecisionTreeClassifier(max_depth=8,min_samples_split=16,min_samples_leaf=32,criterion="gini").fit(egitimgirdi,egitimcikti)

#Eğitim setiyle model kurulması

a = a.fit(egitimgirdi,egitimcikti)

#Kurulan model'in test edilmesi

y_pred = a.predict(valgirdi)

#Çıkan doğruluk skoru ve Hata Matrisi



cm = confusion_matrix(valcikti, y_pred) 

print("Decision Tree confusion_matrix:\n", cm)

print("Decision Tree accuracy_score: ", accuracy_score(valcikti, y_pred)),

print("\nDecision Tree f1_score:",f1_score(valcikti, y_pred)),

filename = 'traditionalml.sav'

pickle.dump(a, open(filename, 'wb'))

print("\n")
a=RandomForestClassifier(n_estimators=250,max_depth=16,min_samples_split=10,max_features="sqrt").fit(egitimgirdi,egitimcikti)

#Eğitim setiyle model kurulması

a = a.fit(egitimgirdi,egitimcikti)

#Kurulan model'in test edilmesi

y_pred = a.predict(valgirdi)

#Çıkan doğruluk skoru ve Hata Matrisi



cm = confusion_matrix(valcikti, y_pred) 

print("Random Forest confusion_matrix:\n", cm)

print("Random Forest accuracy_score: ", accuracy_score(valcikti, y_pred)),

print("\nRandom Forest f1_score:",f1_score(valcikti, y_pred)),

filename = 'traditionalml.sav'

pickle.dump(a, open("ram_model.pkl", 'wb'))

a=pickle.load(open("ram_model.pkl","rb"))

print("\n")
a=AdaBoostClassifier(learning_rate=0.1,n_estimators=120).fit(egitimgirdi,egitimcikti)

#Eğitim setiyle model kurulması

a = a.fit(egitimgirdi,egitimcikti)

#Kurulan model'in test edilmesi

y_pred = a.predict(valgirdi)

#Çıkan doğruluk skoru ve Hata Matrisi



cm = confusion_matrix(valcikti, y_pred) 

print("AdaBoost confusion_matrix:\n", cm)

print("AdaBoost accuracy_score:", accuracy_score(valcikti, y_pred)),

print("\nAdaBoost f1_score:",f1_score(valcikti, y_pred)),

filename = 'traditionalml.sav'

pickle.dump(a, open(filename, 'wb')) 

print("\n")
a=GaussianNB().fit(egitimgirdi,egitimcikti)

#Eğitim setiyle model kurulması

a = a.fit(egitimgirdi,egitimcikti)

#Kurulan model'in test edilmesi

y_pred = a.predict(valgirdi)

#Çıkan doğruluk skoru ve Hata Matrisi



cm = confusion_matrix(valcikti, y_pred) 

print("Naive Bayes confusion_matrix:\n", cm)

print("Naive Bayes accuracy_score: ", accuracy_score(valcikti, y_pred)),

print("\nNaive Bayes f1_score:",f1_score(valcikti, y_pred)),

filename = 'traditionalml.sav'

pickle.dump(a, open(filename, 'wb')) 