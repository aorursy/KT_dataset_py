import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
loans = pd.read_csv('../input/loan_data.csv')
loans.info()
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

                                              bins=30,label='Paid =1')

loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',

                                              bins=30,label='Not Paod=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(11,7))

sns.countplot(x=loans['purpose'] , hue=loans['not.fully.paid'], palette='Set1')
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
sns.lmplot(x='fico', y='int.rate', data=loans, hue='credit.policy' , col='not.fully.paid')
loans.info()
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats , drop_first=True)
final_data.head()
X = final_data.drop('not.fully.paid' , axis=1)

y = final_data['not.fully.paid']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y , test_size=0.3 , random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pre = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pre))
print(confusion_matrix(y_test,y_pre))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=900)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))