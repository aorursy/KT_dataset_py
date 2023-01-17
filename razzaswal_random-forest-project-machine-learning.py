import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
loans = pd.read_csv('../input/loan-dataset/loan_data.csv')
loans.head()
loans.info()
loans.describe()
sns.set()
sns.countplot('credit.policy', data=loans)
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='credit.policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='credit.policy=0')
plt.legend()
plt.xlabel('FICO')

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')

plt.figure(figsize=(12,12))
sns.countplot('purpose', hue='not.fully.paid', data=loans)

sns.jointplot(x='fico', y= 'int.rate', data= loans)

sns.lmplot(x='fico', y='int.rate', col='not.fully.paid', hue='credit.policy', data=loans)

loans.info()
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.head()
final_data.info()
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score
accuracy_score(predictions, y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(predictions, y_test)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
from sklearn.metrics import accuracy_score
accuracy_score(predictions, y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(predictions, y_test)