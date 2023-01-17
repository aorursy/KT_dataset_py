# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split



from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import classification_report,confusion_matrix



from sklearn.ensemble import RandomForestClassifier
dfloan=pd.read_csv("../input/loan_data.csv")

dfloan.head()
dfloan.info()
dfloan.describe()
plt.figure(figsize=(15,10))

dfloan[dfloan['credit.policy']==1]['fico'].hist(bins=30,color='blue',rwidth=0.95,label='Credit Policy=1',alpha=0.6)

dfloan[dfloan['credit.policy']==0]['fico'].hist(bins=30,color='grey',rwidth=0.95,label='credit policy=0',alpha=0.6)

plt.legend()

plt.xlabel('FICO')



plt.figure(figsize=(15,10))

dfloan[dfloan['not.fully.paid']==1]['fico'].hist(bins=30,color='blue',label='Not Fully Paid=1',alpha=0.6)

dfloan[dfloan['not.fully.paid']==0]['fico'].hist(bins=30,color='grey',label='Not Fully Paid=0',alpha=0.6)

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(12,8))

sns.countplot(x='purpose',hue='not.fully.paid',data=dfloan,palette='Set1')
sns.jointplot(x='fico',y='int.rate',data=dfloan)
plt.figure(figsize=(11,7))

sns.lmplot(y='int.rate',x='fico',data=dfloan,hue='credit.policy',col='not.fully.paid',palette='Set1')
cat_feats=['purpose']
final_data=pd.get_dummies(dfloan,columns=cat_feats,drop_first=True)
final_data.info()
final_data.head()
X=final_data.drop('not.fully.paid',axis=1)

y=final_data['not.fully.paid']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dtreepredictions=dtree.predict(X_test)
print(classification_report(y_test,dtreepredictions))
print(confusion_matrix(y_test,dtreepredictions))
rfc=RandomForestClassifier(n_estimators=300)
rfc.fit(X_train,y_train)
rfcpredictions=rfc.predict(X_test)
print(classification_report(y_test,rfcpredictions))
print(confusion_matrix(y_test,rfcpredictions))