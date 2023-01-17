# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
loan=pd.read_csv('../input/predicting-who-pays-back-loans/loan_data.csv')
loan.head(2)
loan.shape
loan.info(verbose=True)
loan.isnull().sum()
loan.describe()
#Here we do exploratry data analysis:-

plt.figure(figsize=(10,6))

loan[loan['credit.policy']==1]['fico'].hist(alpha=0.5,color='green',bins=30,label='Credit.Policy=1')

loan[loan['credit.policy']==1]['fico'].hist(alpha=0.5,color='y',bins=50,label='Credit.Policy=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(10,6))

loan[loan['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='r',bins=30,label='not.fully.paid=1')

loan[loan['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='g',bins=50,label='not.fully.paid=0')

plt.legend()

plt.xlabel('FICO')
plt.figure(figsize=(15,15))

sns.countplot(x='purpose',hue='not.fully.paid',palette='Set1',data=loan)
sns.jointplot(data=loan,x='fico',y='int.rate',color='g')
sns.lmplot(x='fico',y='int.rate',col='not.fully.paid',hue='credit.policy',palette="Set1",data=loan,aspect=1.5)
loan['purpose'].nunique()
feature=['purpose']

#used duummy varible to convert categorical data to numerical data

loans=pd.get_dummies(loan,columns=feature,drop_first=True)

loans.head()
loans.shape
loans.info(verbose=True)
x=loans.drop(['not.fully.paid'],axis=1)

y=loans['not.fully.paid']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
from sklearn import svm

clf=svm.SVC(kernel='rbf',gamma='auto',C=1.0)

clf.fit(X_train,y_train)

clf.fit(X_test,y_test)

trainscore=clf.score(X_train,y_train)

print('trainscore',trainscore)

testscore=clf.score(X_test,y_test)

print('testscore',testscore)

svcpredict=clf.predict(X_test)

print('svc predict',svcpredict)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,svcpredict))

print(confusion_matrix(y_test,svcpredict))