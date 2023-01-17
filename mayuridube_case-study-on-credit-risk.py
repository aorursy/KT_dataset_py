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
#load the train dataset

train = pd.read_csv('../input/loanprediction/train_ctrUa4K.csv')

# display first few rows

display(train.head())
# import and display test dataset

test = pd.read_csv('../input/loanprediction/test_lAUu6dG.csv')

display(test.head())
print(train.shape,test.shape)
# check data type,null values

print(train.info())
# from above info we able to know our dataset having null values

# Also having categorical data

# checking occurance of that data

train.describe(include=['O'])
# importing necessary package



import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Image, display

%matplotlib inline



# Analyzed features Using histogram for it

applicant_income = sns.FacetGrid(train, col='Loan_Status')

applicant_income.map(plt.hist, 'Gender', bins=5)
married = sns.FacetGrid(train, col='Loan_Status')

married.map(plt.hist, 'Married', bins=5)
married = sns.FacetGrid(train, col='Loan_Status')

married.map(plt.hist, 'Self_Employed')
credit_history = sns.FacetGrid(train, col='Loan_Status')

credit_history.map(plt.hist, 'Credit_History')
property_area = sns.FacetGrid(train, col='Loan_Status')

property_area.map(plt.hist, 'Property_Area')
loan_amt = sns.FacetGrid(train, col='Loan_Status')

loan_amt.map(plt.hist, 'Property_Area')
# Make copy of dataset

# we are dropping Loan_ID from our dataset

train_m = train.drop(['Loan_ID'], axis=1)

test_m = test.drop(['Loan_ID'], axis=1)

# print new shape

train_m.shape, test_m.shape
# Now fill null values 

# get the values and features having null values

print(train_m.isnull().sum())

print(test_m.isnull().sum())
# replace null value in gender with most frequent value i.e 'male'

train_m['Gender'] = train_m['Gender'].fillna(train_m['Gender'].dropna().mode().values[0] )

test_m['Gender'] = test_m['Gender'].fillna(test_m['Gender'].dropna().mode().values[0] )



# replace null value in Married with most frequent value i.e 'yes'

train_m['Married'] = train_m['Married'].fillna(train_m['Married'].dropna().mode().values[0] )



# doing same for dependent and self_employed

train_m['Dependents'] = train_m['Dependents'].fillna(train_m['Dependents'].dropna().mode().values[0])

test_m['Dependents'] = test_m['Dependents'].fillna(test_m['Dependents'].dropna().mode().values[0])



train_m['Self_Employed'] = train_m['Self_Employed'].fillna(train_m['Self_Employed'].dropna().mode().values[0])

test_m['Self_Employed'] = test_m['Self_Employed'].fillna(test_m['Self_Employed'].dropna().mode().values[0])
# Replacing Loan_Amount_Term and Credit_History by mode too



train_m['Loan_Amount_Term'] = train_m['Loan_Amount_Term'].fillna(train_m['Loan_Amount_Term'].dropna().mode().values[0])

test_m['Loan_Amount_Term'] = test_m['Loan_Amount_Term'].fillna(test_m['Loan_Amount_Term'].dropna().mode().values[0])



train_m['Credit_History'] = train_m['Credit_History'].fillna(train_m['Credit_History'].dropna().mode().values[0])

test_m['Credit_History'] = test_m['Credit_History'].fillna(test_m['Credit_History'].dropna().mode().values[0])
# replacing by median

train_m['LoanAmount'] = train_m['LoanAmount'].fillna(train_m['LoanAmount'].dropna().median())

test_m['LoanAmount'] = test_m['LoanAmount'].fillna(test_m['LoanAmount'].dropna().median())
# Again check for null values

print(train_m.info())
# now need to convert object data type into numerical category

from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

train_m['Gender'] = lb_make.fit_transform(train_m['Gender'])

test_m['Gender'] = lb_make.fit_transform(test_m['Gender'])





train_m.head()

test_m.head()
train_m['Married'] = lb_make.fit_transform(train_m['Married'])

test_m['Married'] = lb_make.fit_transform(test_m['Married'])
train_m['Education'] = lb_make.fit_transform(train_m['Education'])

test_m['Education'] = lb_make.fit_transform(test_m['Education'])
train_m['Self_Employed'] = lb_make.fit_transform(train_m['Self_Employed'])

test_m['Self_Employed'] = lb_make.fit_transform(test_m['Self_Employed'])
train_m['Property_Area'] = lb_make.fit_transform(train_m['Property_Area'])

test_m['Property_Area'] = lb_make.fit_transform(test_m['Property_Area'])
train_m['Dependents'].value_counts()
train_m.info()

train_m['Dependents'] = train_m['Dependents'].replace('3+','3')

test_m['Dependents'] = test_m['Dependents'].replace('3+','3')
train_m['Dependents'].value_counts()

# train_m.info()
train_m['Dependents'] = pd.to_numeric(train_m['Dependents'])

test_m['Dependents'] = pd.to_numeric(test_m['Dependents'])
train_m.info()
test_m.info()
train_m['Loan_Status'] = lb_make.fit_transform(train_m['Loan_Status'])
# Heatmap: Showing the correlations of features with the target. No correlations are extremely high.

# The correlations between LoanAmount and ApplicantIncome can be explained.

sns.heatmap(train_m.corr())
# Separating dependent variable and feature variable list

y = train_m['Loan_Status']

x = train_m.drop('Loan_Status', axis = 1)
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
model = LogisticRegression()

model.fit(x_train, y_train)

ypred = model.predict(x_test)

evaluation = f1_score(y_test, ypred)

evaluation
ypred_test = model.predict(test_m)

print(ypred_test)