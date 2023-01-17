# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train=pd.read_csv('../input/loan-prediction-analyticsvidhya/train_ctrUa4K.csv')
train.head()
test=pd.read_csv("../input/loan-prediction-analyticsvidhya/test_lAUu6dG.csv")
test.head()
train.isnull().sum()
train['LoanAmount']=train['LoanAmount'].fillna(train['LoanAmount'].mean())
test['LoanAmount']=test['LoanAmount'].fillna(test['LoanAmount'].mean())
train['Loan_Amount_Term']=train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean())
test['Loan_Amount_Term']=test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mean())
a=train[train['Gender']=='Male']['ApplicantIncome'].mean()

a
train['Married'].value_counts()
train['Married'].fillna('Yes',inplace=True)
test['Married'].fillna('Yes',inplace=True)
train
train['Self_Employed'].fillna('Yes',inplace=True)
test['Self_Employed'].fillna('Yes',inplace=True)
train['Dependents'].value_counts()
train['Dependents'].replace('3+',3,inplace=True)
test['Dependents'].replace('3+',3,inplace=True)
train['Dependents'].value_counts()
train['Dependents']=train['Dependents'].astype('float64')
test['Dependents']=test['Dependents'].astype('float64')
train['Dependents']=train['Dependents'].fillna(train['Dependents'].mean())
test['Dependents']=test['Dependents'].fillna(test['Dependents'].mean())
test.isnull().sum()
train['Credit_History']
train['Credit_History'].value_counts()
train['Property_Area'].value_counts()
from sklearn.preprocessing import LabelEncoder
l1=LabelEncoder()
train['Property_Area']=l1.fit_transform(train['Property_Area'])
test['Property_Area']=l1.fit_transform(test['Property_Area'])
train['Gender']=train['Gender'].fillna('Male')
test['Gender']=test['Gender'].fillna('Male')
train.isnull().sum()
test.isnull().sum()
train['Credit_History']=train['Credit_History'].fillna(-1)
test['Credit_History']=test['Credit_History'].fillna(-1)
test.isnull().sum()
train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
l2=LabelEncoder()
l3=LabelEncoder()
l4=LabelEncoder()
train['Gender']=l2.fit_transform(train['Gender'])
test['Gender']=l2.fit_transform(test['Gender'])
train['Married']=l2.fit_transform(train['Married'])
test['Married']=l2.fit_transform(test['Married'])
train['Education']=l3.fit_transform(train['Education'])
test['Education']=l3.fit_transform(test['Education'])

train['Self_Employed']=l4.fit_transform(train['Self_Employed'])
test['Self_Employed']=l4.fit_transform(test['Self_Employed'])
sc=StandardScaler()
X=train.drop(['Loan_Status','Loan_ID'],axis=1)
Y=train['Loan_Status'].map({'Y':1,'N':0})
X=sc.fit_transform(X)
tst=sc.fit_transform(test.drop('Loan_ID',axis=1))
X
Y=Y.values
Y
from sklearn.metrics import classification_report
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
model1=LogisticRegression()
model1.fit(X_train,Y_train)

model1.score(X_train,Y_train)

model2=SVC()
model2.fit(X_train,Y_train)
model2.score(X_train,Y_train)
model3=RandomForestClassifier()
model3.fit(X_train,Y_train)
model3.score(X_train,Y_train)
from sklearn import metrics
yp=model3.predict(X_test)
metrics.f1_score(yp,Y_test)
metrics.confusion_matrix(yp,Y_test)
model4=DecisionTreeClassifier()
model4.fit(X_train,Y_train)
model4.score(X_train,Y_train)
yp1=model4.predict(X_test)
metrics.f1_score(yp1,Y_test)
metrics.confusion_matrix(yp1,Y_test)
from sklearn.ensemble import AdaBoostClassifier
model5=AdaBoostClassifier()
model5.fit(X_train,Y_train)
print(model5.score(X_train,Y_train))
yp2=model5.predict(X_test)
print(metrics.f1_score(yp2,Y_test))
print(metrics.confusion_matrix(yp2,Y_test))
yp2=model5.predict(tst)
smp=pd.read_csv('../input/loan-prediction-analyticsvidhya/sample_submission_49d68Cx.csv')
yp2=pd.DataFrame(yp2)
yp2=yp2[0].map({1:'Y',0:'N'})
smp['Loan_Status']=yp2
smp
from keras.models import Sequential
from keras.layers import Dense,Dropout
model=Sequential()
model.add(Dense(11,'relu',input_shape=(11,)))
model.add(Dense(16,'relu'))
model.add(Dense(32,'relu'))
model.add(Dropout(0.1))
model.add(Dense(32,'relu'))
model.add(Dense(16,'relu'))
model.add(Dense(4,'relu'))
model.add(Dense(1,'sigmoid'))
model.compile('adam','binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=100,batch_size=1)
smp.to_csv('av.csv',index=False)
