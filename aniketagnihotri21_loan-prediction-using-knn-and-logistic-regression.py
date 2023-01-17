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

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
path_train = '/kaggle/input/loan-pred-traincsv/'

    

pl_train = pd.read_csv(path_train+'Loan pred_train.csv')

pl_train.head()
pl_train.columns
pl_train.info()
pl_train.isnull().sum()
# Running value counts on Gender for train

pl_train['Gender'].value_counts()
pl_train['Gender'] = pl_train['Gender'].fillna('unknown')
# Value counts after fillna

pl_train['Gender'].value_counts()
pl_train['Married'].value_counts()
pl_train['Married'] = pl_train['Married'].fillna('unknown')
# Value counts of Married after fillna

pl_train['Married'].value_counts()
# Running value counts on Dependents for train

pl_train['Dependents'].value_counts()
# FIlling max count value

pl_train['Dependents'] = pl_train['Dependents'].fillna('0')
# chaning 3+ to 3 for ease in future processing

pl_train['Dependents'] = pl_train['Dependents'].replace({'3+':3})
# Value Counts after fillna and chaning 3+ to 3

pl_train['Dependents'].value_counts()
pl_train['Self_Employed'].value_counts()
pl_train['Self_Employed'] = pl_train['Self_Employed'].fillna('unknown')
# Running Value conuts again after fillna

pl_train['Self_Employed'].value_counts()
np.mean(pl_train['Loan_Amount_Term'])
# Replacing null values with mean of data

pl_train['Loan_Amount_Term'] = pl_train['Loan_Amount_Term'].fillna(342.0)
pl_train['Loan_Amount_Term'].isnull().sum()
# Running value counts on Credit_History for train

pl_train['Credit_History'].value_counts()
# Creating additional category

pl_train['Credit_History'] = pl_train['Credit_History'].fillna(1.0)
# Running value counts on Loan Amount Terms

pl_train['Credit_History'].value_counts()
np.mean(pl_train['LoanAmount'])
#filling null values of Loan amount with mean(146.4)

pl_train['LoanAmount'] = pl_train['LoanAmount'].fillna(146.4)
pl_train['LoanAmount'].isnull().sum()
pl_train.isnull().sum()
pl_train['Property_Area'].value_counts()
mylist_train = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area' , 'Loan_Status']
for i in mylist_train:

    pl_train[i] = pl_train[i].astype({i:'category'})
pl_train.info()
# Evaluating percentage of loan status to yes and no

pl_train['Loan_Status'].value_counts(normalize=True)*100
sns.factorplot('Loan_Status','ApplicantIncome', data=pl_train, hue='Self_Employed')
sns.factorplot('Loan_Status','ApplicantIncome', data=pl_train,hue='Credit_History')
sns.factorplot('Loan_Status','ApplicantIncome', data=pl_train,hue='Education')
sns.factorplot('Loan_Status','ApplicantIncome', data=pl_train, hue='Property_Area')
sns.factorplot('Loan_Status','ApplicantIncome', data=pl_train, hue='Dependents')
sns.factorplot('Loan_Status','CoapplicantIncome', data=pl_train, hue='Married')
sns.factorplot('Loan_Status','CoapplicantIncome', data=pl_train, hue='Dependents')
# Male = 0, Female = 1

pl_train['Gender'] = pl_train['Gender'].replace({'Male':0, 'Female':1,'unknown' : 2})
# Yes = 1, No = 0

pl_train['Married'] = pl_train['Married'].replace({'Yes' :1, 'No': 0, 'unknown':2})
# Graduate = 1, Not Graduate = 0

pl_train['Education'] = pl_train['Education'].replace ({'Graduate' : 1, 'Not Graduate' : 0})
# Yes: 1

# No : 0

# unknown:2

pl_train['Self_Employed'] = pl_train['Self_Employed'].replace ({'Yes': 1,'No' : 0, 'unknown':2})
pl_train['Property_Area'] = pl_train['Property_Area'].replace ({'Semiurban': 1,'Urban' : 0, 'Rural':2})
pl_train['Loan_Status'] = pl_train['Loan_Status'].replace({'Y':1, 'N':0})
pl_train = pl_train.drop(columns=['Loan_ID'])
X_train = pl_train.drop(columns=['Loan_Status'])
y_train = pl_train['Loan_Status']
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_train,y_train)
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(X_train,y_train)