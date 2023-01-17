import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
loan_data = pd.read_csv("loan_data.csv")
loan_data.head()
loan_data.describe()
plt.figure(figsize=(15,9))

loan_data[loan_data['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',

                                              bins=30,label='Credit.Policy=1')

loan_data[loan_data['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',

                                              bins=30,label='Credit.Policy=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(15,9))

loan_data[loan_data['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',

                                              bins=30,label='not.fully.paid=1')

loan_data[loan_data['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',

                                              bins=30,label='not.fully.paid=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(15,9))

sns.countplot(x="purpose",hue='not.fully.paid',data=loan_data, palette="Paired")
sns.set(style="whitegrid")

sns.jointplot(x='fico', y='int.rate', data=loan_data, color="purple")
# call regplot on each axes

#fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

sns.set(style="darkgrid")

sns.lmplot("fico", "int.rate", data=loan_data,hue= 'credit.policy',col='not.fully.paid', palette= 'Set1')

#sns.regplot(x=idx, y=df['x'], ax=ax1)

#sns.regplot(x=idx, y=df['y'], ax=ax2)
loan_data.info()
purpose = pd.get_dummies(loan_data['purpose'],drop_first=True)

loan_data.drop(['purpose'],axis=1,inplace=True)
final_data = pd.concat([loan_data,purpose],axis=1)
final_data.head()
from sklearn.model_selection import train_test_split
X = loan_data.drop('not.fully.paid',axis=1)

y = loan_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))