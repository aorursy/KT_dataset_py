import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
loans = pd.read_csv('../input/loandata/loan_data.csv')
loans.info()
loans.describe()
loans.head()
plt.figure(figsize=(12,6))
loans[loans['credit.policy']==1]['fico'].hist(bins=30)
loans[loans['credit.policy']==0]['fico'].hist(bins=30)
plt.legend(['credit policy - 1','credit policy - 0'])
plt.xlabel('FICO')
plt.figure(figsize=(12,6))
loans[loans['not.fully.paid']==0]['fico'].hist(bins=30)
loans[loans['not.fully.paid']==1]['fico'].hist(bins=30)
plt.legend(['not.fully.paid - 1','not.fully.paid - 0'])
plt.xlabel('FICO')
plt.figure(figsize=(12,6))
sns.countplot(x='purpose' , hue='not.fully.paid',data=loans)
plt.tight_layout()
sns.jointplot(x='fico',y='int.rate',data=loans)
plt.figure(figsize=(12,6))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')
loans.info()
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.info()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_data.drop('not.fully.paid',axis=1), final_data['not.fully.paid'], test_size=0.30, random_state=101)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model = model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print('*'*30)
print(classification_report(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1400)
rf.fit(X_train,y_train)
rfPredict = rf.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,rfPredict))
print(confusion_matrix(y_test,rfPredict))
