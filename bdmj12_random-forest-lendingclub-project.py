import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
loans = pd.read_csv('../input/loan_data.csv')
loans.head()

loans.info()
loans.describe()
plt.figure(figsize=(11,7))
sns.countplot(loans['purpose'], hue = loans['not.fully.paid'])
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
sns.pairplot(loans.drop(['credit.policy', 'purpose',
       'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid'], axis=1))
final_data = pd.get_dummies(loans,columns = ['purpose'], drop_first=True)
final_data.head()
from sklearn.model_selection import train_test_split
X= final_data.drop('not.fully.paid', axis=1)
y= final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 101)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)
np.array((pred==y_test)).sum()
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
from sklearn.ensemble import RandomForestClassifier
dfor = RandomForestClassifier()
dfor.fit(X_train, y_train)
pred2 = dfor.predict(X_test)
(y_test == pred2).sum()
print(classification_report(y_test,pred2))
print(confusion_matrix(y_test,pred2))
