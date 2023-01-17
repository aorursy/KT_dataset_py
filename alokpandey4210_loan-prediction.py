# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/loan-prediction-dataset/train.csv')

df.head()
df.corr(method='pearson')
X=df.drop(['Loan_ID','Loan_Status'],axis=1)
X.head()
y=df['Loan_Status']
X.isnull().sum()
X['Gender'].value_counts()
X['Gender'].fillna('Male',inplace=True)
X['Married'].value_counts()
X['Married'].fillna('Yes',inplace=True)
X['Dependents'].value_counts()
X['Dependents'].fillna(0,inplace=True)
X['Self_Employed'].value_counts()
X['Self_Employed'].fillna('No',inplace=True)
X['LoanAmount'].value_counts()
X['LoanAmount'].fillna(X['LoanAmount'].mean(),inplace=True)
X['Loan_Amount_Term'].value_counts()
X['Loan_Amount_Term'].fillna(X['Loan_Amount_Term'].mean(),inplace=True)
X['Credit_History'].value_counts()
X['Credit_History'].fillna(1,inplace=True)
from sklearn.preprocessing import LabelBinarizer

lb=LabelBinarizer()

y=lb.fit_transform(y)

y
X['ApplicantIncome'].hist(bins=50)
X.boxplot(column='ApplicantIncome',by='LoanAmount')
X['ApplicantIncome'].plot('density',color='Red')
X['CoapplicantIncome'].hist(bins=50)
X['CoapplicantIncome'].plot('density')
X['LoanAmount'].hist(bins=50)
X['LoanAmount'].plot('density',color='Green')
X.boxplot(column='LoanAmount',by='Credit_History')
X=pd.get_dummies(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
from sklearn.linear_model import LogisticRegression

f=LogisticRegression()

f.fit(X_train,y_train)

f.score(X_test,y_test)
from sklearn.ensemble import ExtraTreesClassifier

j=ExtraTreesClassifier()

j.fit(X_train,y_train)

j.score(X_test,y_test)
from sklearn.linear_model import SGDClassifier

m=SGDClassifier()

m.fit(X_train,y_train)

m.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier

l=DecisionTreeClassifier()

l.fit(X_train,y_train)

l.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

s=RandomForestClassifier()

s.fit(X_train,y_train)

s.score(X_test,y_test)
from sklearn.naive_bayes import GaussianNB

y=GaussianNB()

y.fit(X_train,y_train)

y.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

q=KNeighborsClassifier()

q.fit(X_train,y_train)

q.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

y_predict=f.predict(X_test)

results=confusion_matrix(y_test,y_predict)



print(results)

# saving model

import pickle 

file_name='Loan.sav'

tuples=(f,X)

pickle.dump(tuples,open(file_name,'wb'))
from sklearn.metrics import confusion_matrix

y_predict1=j.predict(X_test)

results=confusion_matrix(y_test,y_predict)

print(results)

from sklearn.metrics import confusion_matrix

y_predict1=f.predict(X_test)

results1=confusion_matrix(y_test,y_predict1)

print(results1)

from sklearn.metrics import confusion_matrix

y_predict21=j.predict(X_test)

results12=confusion_matrix(y_test,y_predict21)

print(results12)

from sklearn.metrics import confusion_matrix

y_predict2=m.predict(X_test)

results2=confusion_matrix(y_test,y_predict2)

print(results2)

from sklearn.metrics import confusion_matrix

y_predict3=l.predict(X_test)

results3=confusion_matrix(y_test,y_predict3)

print(results3)

from sklearn.metrics import confusion_matrix

y_predict4=s.predict(X_test)

results4=confusion_matrix(y_test,y_predict4)

print(results4)

from sklearn.metrics import confusion_matrix

y_predict5=y.predict(X_test)

results5=confusion_matrix(y_test,y_predict5)

print(results5)

from sklearn.metrics import roc_curve,auc,roc_auc_score

fpr,tpr,threshold=roc_curve(y_test,j.predict_proba(X_test)[:,1])
fpr
tpr
threshold
fpr1,tpr1,threshold1=roc_curve(y_test,f.predict_proba(X_test)[:,1])
fpr1
tpr1
threshold1
fpr2,tpr2,threshold2=roc_curve(y_test,l.predict_proba(X_test)[:,1])
fpr2
tpr2
threshold2
fpr3,tpr3,threshold3=roc_curve(y_test,s.predict_proba(X_test)[:,1])
fpr3
tpr3
threshold3
fpr4,tpr4,threshold4=roc_curve(y_test,y.predict_proba(X_test)[:,1])
fpr4
tpr4
threshold4
roc_auc=roc_auc_score(y_test,j.predict_proba(X_test)[:,1])

roc_auc
roc_auc1=roc_auc_score(y_test,f.predict_proba(X_test)[:,1])
roc_auc1
roc_auc2=roc_auc_score(y_test,l.predict_proba(X_test)[:,1])
roc_auc2
roc_auc3=roc_auc_score(y_test,s.predict_proba(X_test)[:,1])
roc_auc3
roc_auc4=roc_auc_score(y_test,y.predict_proba(X_test)[:,1])
roc_auc4
y_PRED=f.predict_proba(X_test)
y_PRED
from sklearn.preprocessing import binarize



y_pred_c=binarize(y_PRED)

y_pred_c
yAlok=y_pred_c[:,1]
y_AAA=yAlok.astype(int)
from sklearn.metrics import confusion_matrix

qq=confusion_matrix(y_test,y_AAA)

qq
from sklearn.preprocessing import binarize



y_pred_class1=binarize(y_PRED,0.60)

y_pred_class1
yFETCH=y_pred_class1[:,1]
yFETCH1=yFETCH.astype(int)
from sklearn.metrics import confusion_matrix

resultq=confusion_matrix(y_test,yFETCH1)
resultq
from sklearn.preprocessing import binarize



y_pred_class1=binarize(y_PRED,0.70)

y_pred_class1
yFETCH=y_pred_class1[:,1]
yFETCH1=yFETCH.astype(int)
from sklearn.metrics import confusion_matrix

resultw=confusion_matrix(y_test,yFETCH1)

resultw
from sklearn.preprocessing import binarize



y_pred_class1=binarize(y_PRED,0.80)

y_pred_class1
yFETCH=y_pred_class1[:,1]
yFETCH1=yFETCH.astype(int)
from sklearn.metrics import confusion_matrix

resultp=confusion_matrix(y_test,yFETCH1)

resultp