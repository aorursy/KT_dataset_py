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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
#loading the customer data, buy scooter and the test data

customer = pd.read_csv('/kaggle/input/dscmeetup3/CustomerInfo.csv')

buyscooter = pd.read_csv('/kaggle/input/dscmeetup3/buyscooter.csv')

testdata = pd.read_csv('/kaggle/input/dscmeetup3/testdata.csv')

test_copy = testdata.copy()
customer.head()
buyscooter.head()
testdata.head()


print('we have', customer.shape[0], 'rows and', customer.shape[1], 'columns in Customer Info')

print("==========================================================")

print('we have', buyscooter.shape[0], 'rows and', buyscooter.shape[1], 'columns in Buy scooter data')

print("==========================================================")

print('we have', testdata.shape[0], 'rows and', testdata.shape[1], 'columns in test data')
merged_data = customer.merge(buyscooter)
merged_data.head()
#Create checking point

df = merged_data.copy()
df.columns
df.describe().transpose()
df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female'})
df.Gender.value_counts()


df['MaritalStatus'] = df.MaritalStatus.replace({'M': 'Married', 'S': 'Single'})
df.MaritalStatus.value_counts()
df.Education.value_counts()

#print('------------------------'),

#df.Occupation.value_counts()
object_data = df.dtypes[df.dtypes == 'object'].count()

categorical_data = df.dtypes[df.dtypes == 'int64'].count()

continuous_data = df.dtypes[df.dtypes == 'float64'].count()
print('we have {} object data'.format(object_data))

print('we have {} categorical data'.format(categorical_data))

print('we have {} continuous data'.format(continuous_data))
categorical_features = df.dtypes[df.dtypes == 'object'].index

continuous_features = df.dtypes[df.dtypes == 'int64'].index
# Counts on categorical columns

for feature in categorical_features:

    print(feature,':')

    print(df[feature].value_counts())

    print('----------------------------')
#Columns list

df.columns
# Columns to drop

target = df['BuyScooter']

train_test = [df, testdata]

drop_col = ['CustomerID', 'FirstName', 'MiddleName', 'LastName','City',

       'StateProvinceName','PostalCode', 'PhoneNumber']
for dataset in train_test:

    dataset.drop(drop_col, axis=1, inplace = True)
df.columns
df.dtypes
for dataset in train_test:

    dataset['BirthDate'] =  pd.to_datetime(dataset['BirthDate'])
df.dtypes
testdata.dtypes
testdata.head()
# Calculating Age
for dataset in train_test:

        dataset['TotalAsset'] = dataset['HomeOwnerFlag']+dataset['NumberCarsOwned']
for dataset in train_test:

    if (dataset['TotalChildren']==0).all():

        dataset['ChildIncomeR2'] = dataset['YearyIncome']/1.0

    else:

        dataset['ChildIncomeR2'] = dataset['YearlyIncome']/dataset['TotalChildren']
df['Education'] = df['Education'].replace({'Partial High School': 1, 'High School':2, 'Partial College':3, 'Bachelors':4, 'Bachelors ':4,'Graduate Degree':5}) 

testdata['Education'] = testdata['Education'].replace({'Partial High School': 1, 'High School':2, 'Partial College':3, 'Bachelors':4, 'Bachelors ':4, 'Graduate Degree':5})

train = df.copy()

test = testdata.copy()
testdata.head()
df.TotalChildren.describe()
# from datetime import date

# def calculate_age(born):

#     today = datetime.date.today()

#     return today.year - born - (today.month, today.day) < (born.month, born.day)



# df['Age'] = df['BirthDate'].apply(calculate_age)



# # for dataset in train_test:

# #     dataset['Age'] = dataset['BirthDate'].apply(calculate_age)
categorical_features = df.dtypes[df.dtypes == 'object'].index

continuous_features = df.dtypes[df.dtypes == 'int64'].index
# Counts on categorical columns

for feature in categorical_features:

    print(feature,':')

    print(df[feature].value_counts())

    print('----------------------------')
df.Education.value_counts()
for dataset in train_test:

    dataset.drop(['BirthDate','ChildIncomeR2'], axis=1, inplace = True)
df.head()
testdata = pd.get_dummies(testdata)

df = pd.get_dummies(df)
df.head()
df.shape, testdata.shape
test.Education.value_counts()
test.columns
testdata.head()
features = df.drop('BuyScooter', axis=1)

target = df.BuyScooter
features.shape, testdata.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.20, random_state=0)
# Logistic Regression

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()

reg.fit(X_train, y_train)

print("Train score: ", reg.score(X_train, y_train))

print("Validation Score :",reg.score(X_test, y_test))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)

dt.fit(X_train, y_train)

print("Train score: ", dt.score(X_train, y_train))

print("Validation Score :",dt.score(X_test, y_test))
merged_data.shape, 
prediction = reg.predict(testdata)



submission = pd.DataFrame({'CustomerID': test_copy['CustomerID'],

                          "BuyScooter": prediction})
# Decision Tree

prediction2 = dt.predict(testdata)



submission2 = pd.DataFrame({'CustomerID': test_copy['CustomerID'],

                          "BuyScooter": prediction2})

submission2.to_csv('Submission4.csv', index=False)
submission.head()
submission.to_csv('Submission.csv', index=False)
!kaggle competitions submissions -c dscmeetup3 -f submission.csv -m "Notes"