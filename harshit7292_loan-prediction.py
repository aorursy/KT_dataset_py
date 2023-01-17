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
df=pd.read_csv("/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")
df.head()
df.describe()
df.isnull().sum()
df['LoanAmount'].hist(bins=50)

df.boxplot(column='ApplicantIncome')

df['LoanAmount'].hist(bins=50)
df.boxplot(column='LoanAmount')
df['Property_Area'].value_counts()
temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print ('Frequency Table for Credit History:') 
print (temp1)
print ('\nProbility of getting loan for each Credit History class:')
print (temp2)




import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")


temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

df.isnull().sum()
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)



df.isnull().sum(0)
df['Gender'].value_counts()
df['Gender'].fillna('Male', inplace=True)
df['Credit_History'].value_counts()
df['Credit_History'].fillna(1,inplace=True)
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna('No',inplace=True)
df.isnull().sum()
df['Dependents'].value_counts()
df['Dependents'].fillna(0,inplace=True)
df.isnull().sum()
df['Married'].value_counts()
df['Married'].fillna('Yes',inplace=True)
X=df[['Credit_History','Gender','Married','Education']]
X=pd.get_dummies(X)
y=df['Loan_Status']

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X,y)
pdt=model.predict(X)
from sklearn.metrics import accuracy_score
print(accuracy_score(pdt,y))
df.head()
import sklearn.model_selection as ms
import sklearn.tree as tree
clf=tree.DecisionTreeClassifier(max_depth=3,random_state=200)
mod=ms.GridSearchCV(clf,param_grid={'max_depth':[4]})
mod.fit(X,y)

mod.best_estimator_
mod.best_score_