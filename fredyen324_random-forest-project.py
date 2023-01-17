import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
loans = pd.read_csv('../input/loan_data.csv')
loans.info()
loans.describe()
loans.head()
plt.figure(figsize=(10,6))

loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',

                                              bins=30,label='Credit.Policy=1')

loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',

                                              bins=30,label='Credit.Policy=0')

plt.legend()

plt.xlabel('FICO')



plt.figure(figsize=(10,6))

loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',

                                              bins=30,label='not.fully.paid=1')

loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',

                                              bins=30,label='not.fully.paid=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(11,7))

sns.countplot(x='purpose',data = loans,hue='not.fully.paid')

sns.jointplot(x='fico',y='int.rate',data=loans)
sns.lmplot(x='fico',y='int.rate',hue='credit.policy',col='not.fully.paid',data=loans)
loans.info()
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_data.drop('not.fully.paid',axis=1), final_data['not.fully.paid'], test_size=0.3)
from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
prediction = dtree.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))
from sklearn.ensemble import RandomForestClassifier
rft = RandomForestClassifier(n_estimators=600)
rft.fit(X_train,y_train)
prediction_rft = rft.predict(X_test)
print(classification_report(y_test,prediction_rft))
confusion_matrix(y_test,prediction_rft)