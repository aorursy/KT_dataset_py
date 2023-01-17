# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import numpy

from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import neighbors 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

test=pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')
train.head()
def tdrop(x):

    x.drop("Loan_ID",axis=1,inplace=True)
tdrop(train)
d={"Loan_Status":{'N':0.0,'Y':1.0}}

train.replace(d,inplace=True)
label=train["Loan_Status"].values

train.drop(["Loan_Status"],axis=1,inplace=True)
import numpy as np

def missvalue(a):

    l=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',

       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',

       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    for i in l:

        if a[i].dtypes!='O':

            med=a[i].median()

            a[i].fillna(med,inplace=True)

        else:

            m=a[i].value_counts().index[0]

            a[i].fillna(m,inplace=True) 
missvalue(train)
def split(a):

    num=a.select_dtypes(include=[np.number]) 

    cat=a.select_dtypes(exclude=[np.number])

    cat=pd.get_dummies(cat)

    return num,cat

    
x,y=split(train)
x.columns.value_counts().sum()
y.columns.value_counts().sum()
x.insert(5,"Loan_Status",label,True)

y.insert(15,"Loan_Status",label,True)
x.head()
correlation=x.corr()
plt.figure(figsize=(8,8))

ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           

plt.show()
correlation=y.corr()
plt.figure(figsize=(12,12))

ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           

plt.show()
def fdrop(m,n):

    m.drop(["ApplicantIncome"],axis=1,inplace=True)

    n.drop(["Self_Employed_No"],axis=1,inplace=True)

    n.drop(["Dependents_0"],axis=1,inplace=True)

    n.drop(["Self_Employed_Yes"],axis=1,inplace=True) 

    return m,n
x,y=fdrop(x,y)
x.drop(["Loan_Status"],axis=1,inplace=True)
y.drop(["Loan_Status"],axis=1,inplace=True)
scaler=StandardScaler()

def scaling(x,y):

     features_scaled=scaler.fit_transform(x.values)

     q=y.values

     vk=np.concatenate((q,features_scaled),axis=1)

     return vk
features=scaling(x,y)
numpy.random.seed(1234)

(x_train,x_test,y_train,y_test) = train_test_split(features,label, train_size=0.75, random_state=42)
C = 1.0

svc = svm.SVC(kernel='linear', C=C)
svc.fit(x_train, y_train)
y_pred =svc.predict(x_test)
accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(classification_report(y_test, y_pred))
svc = svm.SVC(kernel='rbf', C=C)
svc.fit(x_train, y_train)
y_pred =svc.predict(x_test)
accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
clf= DecisionTreeClassifier(random_state=1)

clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
clf = RandomForestClassifier(n_estimators=10, random_state=1)

clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
clf = neighbors.KNeighborsClassifier(n_neighbors=10)

clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
tdrop(test)
missvalue(test)
x,y=split(test)
x,y=fdrop(x,y)
features=scaling(x,y)
features
svc = svm.SVC(kernel='rbf', C=C)
svc.fit(x_train,y_train)
y_pred=svc.predict(features)
print(y_pred)