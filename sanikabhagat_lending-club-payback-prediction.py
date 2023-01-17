import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
loans = pd.read_csv('../input/lending-club-data/loan_data.csv')
loans.head()
loans.head().info()
loans.head().shape
loans.head().describe()
plt.figure(figsize=(10,6))

loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Credit Policy=1')

loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='Credit Policy=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(10,6))

loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Not Fully Paid=1')

loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='Not Fully Paid=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(11,7))

sns.countplot('purpose',hue='not.fully.paid',data=loans,palette='Set1')
sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
plt.figure(figsize=(11,7))

sns.lmplot(x='fico',y='int.rate',data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')
loans.info()
loan_purpose=['purpose']
final_data=pd.get_dummies(loans,columns=loan_purpose,drop_first=True)
# In the above code, drop_first is done to avoid multi-colinearity
final_data.info()
final_data.head()
X = final_data.drop('not.fully.paid',axis=1)

y=final_data['not.fully.paid']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.tree import DecisionTreeClassifier
# Instantiating Decision Tree model (basically creating a decision tree object)
dtree = DecisionTreeClassifier()
# Training or fitting the model on training data
dtree.fit(X_train,y_train)
dtree_predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,dtree_predictions))
print(confusion_matrix(y_test,dtree_predictions))
from sklearn.ensemble import RandomForestClassifier
# Instantiating Random Forest model (basically creating a random forest object)
rfc = RandomForestClassifier(n_estimators=300)
# Training or fitting the model on training data
rfc.fit(X_train,y_train)
rfc_predictions = rfc.predict(X_test)
print(classification_report(y_test,rfc_predictions))
print(confusion_matrix(y_test,rfc_predictions))