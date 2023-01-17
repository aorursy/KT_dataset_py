import pandas as pd

import os



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('../input/salary-gender/original.csv')

train
train.head(3)
train.info()
def cleanupSalary(row):

    salary = row['Salary'].replace('$', '')

    salary = float(salary)    #type casting

    return salary
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
train['cleanSalary'] = train.apply(cleanupSalary, axis = 1)
train.head()
print("Average Salary per gender")

train.groupby('gender')['cleanSalary'].mean()
def fSalary(row):

    if row['gender'] == 'Female':

        return row['cleanSalary']

train['femaleSalaryavg'] = train.apply(fSalary, axis =1)

train.head()
def mSalary(row):

    if row['gender'] == 'Male':

        return row['cleanSalary']

train['MaleSalaryavg'] = train.apply(mSalary, axis =1)

train.head()
train.groupby('Job Title')['cleanSalary', 'MaleSalaryavg', 'femaleSalaryavg'].mean()
train.groupby(['Job Title','City'])['cleanSalary', 'MaleSalaryavg', 'femaleSalaryavg'].mean()
train.describe()
train.info()
train['City'].fillna('London',inplace=True)

train.head()
train.info()
median=train['Latitude'].median()

train['Latitude'].fillna(median,inplace=True)

train.head()
train.info()