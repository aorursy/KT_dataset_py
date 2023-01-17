import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
loans=pd.read_csv('../input/loan_data.csv')
loans.info()
loans.describe()
loans.head()
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
plt.hist(loans[loans['credit.policy']==1]['fico'],alpha=0.5,color='blue',bins=30,label='CREDIT.policy =1')
plt.hist(loans[loans['credit.policy']==0]['fico'],alpha=0.5,color='red',bins=30,label='CREDIT.policy =0')
plt.legend()
plt.xlabel('FICO')
plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
plt.hist(loans[loans['not.fully.paid']==1]['fico'],alpha=0.5,color='black',bins=30,label='not fully paid')
plt.hist(loans[loans['not.fully.paid']==0]['fico'],alpha=0.5,color='red',bins=30,label='fully paid')
plt.legend()
plt.xlabel('FICO')
plt.figure(figsize=(11,6))
sns.set_style('darkgrid')
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='viridis')

sns.jointplot(x='fico',y='int.rate',data=loans,kind='scatter',color='purple',size=8)
plt.figure(figsize=(11,7))
sns.lmplot(x='fico',y='int.rate',data=loans,hue='credit.policy',col='not.fully.paid',palette='plasma')
loans.info()
cat_feats=['purpose']
final_data=pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.head()
X=final_data.drop('not.fully.paid',axis=1)
y=final_data['not.fully.paid']

from sklearn.cross_validation import train_test_split as t
X_train, X_test, y_train, y_test = t(X, y, test_size=0.3, random_state=101)

from sklearn.tree import DecisionTreeClassifier
dtree =DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=800)
rfc.fit(X_train,y_train)
y_pred_rfc = rfc.predict(X_test)
print(classification_report(y_test,y_pred_rfc))
print(confusion_matrix(y_test,y_pred_rfc))