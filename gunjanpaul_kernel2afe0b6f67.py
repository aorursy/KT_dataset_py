import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
loans=pd.read_csv("../input/loan_1.csv")
loans.info()
loans.describe()
loans.head()
plt.figure(figsize=(10,6))

loans[loans['credit.policy']==1]['fico'].hist(bins=35,color='blue',label='credit policy=0',alpha=.7)

loans[loans['credit.policy']==0]['fico'].hist(bins=35,color='red',label='credit policy=0',alpha=.7)

plt.legend()

plt.xlabel('FICO')

plt.figure(figsize=(10,6))

loans[loans['not.fully.paid']==1]['fico'].hist(bins=35,color='blue',label='Not Fully Paid=1',alpha=.7)

loans[loans['not.fully.paid']==0]['fico'].hist(bins=35,color='red',label='Not Fully Paid=0',alpha=.7)

plt.legend()

plt.xlabel('FICO')

plt.figure(figsize=(11,7))

sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
plt.figure(figsize=(13,9))

sns.jointplot(x='fico',y='int.rate',data=loans,color='green')
plt.figure(figsize=(10,8))

sns.lmplot(y='int.rate',x='fico',data=loans,hue="credit.policy",col='not.fully.paid',palette='Set1')
loans.info()
cat_feats=['purpose']
final_data=pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.head()
from sklearn.model_selection import train_test_split
X=final_data.drop('not.fully.paid',axis=1)

y=final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
prediction=dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=300)
rfc.fit(X_train,y_train)
predictions=rfc.predict(X_test)
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))