import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('../input/mushroom-classification/mushrooms.csv')
df.head()
print("The data set has {} rows and {} columns".format(df.shape[0],df.shape[1]))
df.info()
df.isnull().sum().max()
df.describe()
df.columns
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

for col in df.columns:

    df[col]=label.fit_transform(df[col])

df.head()
df['class'].value_counts()
X=df.drop(['class'],axis=1,inplace=False)
y=df['class']
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.neighbors import KNeighborsClassifier
mod=KNeighborsClassifier(n_neighbors=5)

mod.fit(X_train,y_train)
pred=mod.predict(X_test)
print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
from sklearn.linear_model import LogisticRegression
lmodel=LogisticRegression()

lmodel.fit(X_train,y_train)
lm_predict=lmodel.predict(X_test)
confusion_matrix(y_test,lm_predict)
print(classification_report(y_test,lm_predict))
from sklearn.svm import SVC
smodel=SVC()
smodel.fit(X_train,y_train)
spredict=smodel.predict(X_test)
confusion_matrix(y_test,spredict)
print(classification_report(y_test,spredict))
from sklearn.naive_bayes import GaussianNB

nmodel=GaussianNB()
nmodel.fit(X_train,y_train)
npred=nmodel.predict(X_test)
confusion_matrix(y_test,npred)
print(classification_report(y_test,npred))