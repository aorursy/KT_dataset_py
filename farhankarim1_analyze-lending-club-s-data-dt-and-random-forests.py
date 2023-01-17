import pandas as pd
import seaborn as sns
#not complete data it's clean and reduced rows
data = pd.read_csv('../input/loan_data.csv')
data.shape
data.info()
data.columns

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data[['int.rate', 'installment', 'log.annual.inc',
       'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']] = scaler.fit_transform(data[['int.rate', 'installment', 'log.annual.inc',
       'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']])
data
#EDA
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(10,6))
data[data['credit.policy']==1]['fico'].hist(bins=10,color='red',label='CP=1',alpha=0.6)
data[data['credit.policy']==0]['fico'].hist(bins=10,color='green',label='CP=0',alpha=0.6)
plt.legend()
plt.xlabel('FICO SCORE')
#sns.pairplot(data,hue='credit.policy')
plt.figure(figsize=(10,6))
sns.countplot(x='purpose',hue='not.fully.paid',data=data)
sns.jointplot(x='fico',y='int.rate',data=data)
data.columns
#purpose is categorical
cat_purpose = pd.get_dummies(data,columns=['purpose'],drop_first=True)
cat_purpose.info()
from sklearn.cross_validation import train_test_split
X=cat_purpose.drop('not.fully.paid',axis=1)
y=cat_purpose['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.30,random_state=101)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
pred = dt.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
#THAT WEN VERY WELL! FML!!
from sklearn.ensemble import RandomForestClassifier
import numpy as np
error_rate=[]
for i in range(260,280,1):
    rforest = RandomForestClassifier(n_estimators=i)
    rforest.fit(X_train,y_train)
    pred = rforest.predict(X_test)
    print(classification_report(y_test,pred))
    error_rate.append(np.mean(pred != y_test))
    print(confusion_matrix(y_test,pred))
plt.figure(figsize=(10,6))
plt.plot(range(260,280,1),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
rforest = RandomForestClassifier(n_estimators=275)
rforest.fit(X_train,y_train)
pred = rforest.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
