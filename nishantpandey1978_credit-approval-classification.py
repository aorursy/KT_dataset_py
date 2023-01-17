import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16']

dataset=pd.read_csv('../input/credit-approval/crx.data',names=columns,na_values='?')

dataset.isnull().sum()

miss_val_string=['A1','A4','A5','A6','A7']

miss_val_int=['A2','A14']
from sklearn.impute import SimpleImputer

si_str=SimpleImputer(strategy='most_frequent')

si_int=SimpleImputer(strategy='median')
for e in miss_val_string:

    dataset.loc[:,[e]]=si_str.fit_transform(dataset.loc[:,[e]])
for e in miss_val_int:

    dataset.loc[:,[e]]=si_int.fit_transform(dataset.loc[:,[e]])
dataset.isnull().sum()
int_columns=['A2','A3','A8','A11','A14','A15']

from sklearn.preprocessing import MinMaxScaler

mm=MinMaxScaler()
for e in int_columns:

    dataset.loc[:,[e]]=mm.fit_transform(dataset.loc[:,[e]])
X=dataset.drop('A16',axis=1)

y=dataset['A16']
str_columns={e for e in columns if e not in int_columns}    

str_columns.remove('A16')
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
for e in str_columns:

    X[e]=le.fit_transform(X[e])



y=le.fit_transform(y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()
knn.fit(X_train,y_train)

knn.score(X_train,y_train)



y_pred=knn.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report



print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()



log_reg.fit(X_train,y_train)

log_reg.score(X_train,y_train)

y_predlr=log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report



print(confusion_matrix(y_test,y_predlr))

print(classification_report(y_test,y_predlr))
from sklearn.svm import SVC

svm=SVC()



svm.fit(X_train,y_train)

svm.score(X_train,y_train)



y_predsvm=svm.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report



print(confusion_matrix(y_test,y_predsvm))

print(classification_report(y_test,y_predsvm))
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()



nb.fit(X_train,y_train)

nb.score(X_train,y_train)



y_prednb=nb.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report



print(confusion_matrix(y_test,y_prednb))

print(classification_report(y_test,y_prednb))
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()



dt.fit(X_train,y_train)

dt.score(X_train,y_train)



y_preddt=dt.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report



print(confusion_matrix(y_test,y_preddt))

print(classification_report(y_test,y_preddt))
from sklearn.naive_bayes import MultinomialNB

mnb=GaussianNB()



mnb.fit(X_train,y_train)

mnb.score(X_train,y_train)



y_predmnb=mnb.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report



print(confusion_matrix(y_test,y_predmnb))

print(classification_report(y_test,y_predmnb))