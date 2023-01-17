# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample = pd.read_csv('/kaggle/input/GiveMeSomeCredit/sampleEntry.csv')

credit_train = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')

credit_test = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-test.csv')
credit_train.info()
credit_train.describe()
credit_train.head()
sns.pairplot(credit_train)
credit_train.columns
plt.figure(figsize=(15,10))

sns.heatmap(credit_train.isnull(),yticklabels=False,cbar=False)
plt.figure(figsize=(15,10))

sns.jointplot(data = credit_train, x= 'age', y = 'SeriousDlqin2yrs')
sns.distplot(credit_train['age'].dropna(),kde=False,color='darkred',bins=30)
sns.distplot(credit_train['DebtRatio'].dropna(),kde=False,color='darkred',bins=500)
sns.distplot(credit_train['MonthlyIncome'].dropna(),kde=False,color='darkred',bins=30)
credit_train.corr()
credit_train.head()
print(credit_train.isnull().sum())
print(credit_test.isnull().sum())
credit_train['MonthlyIncome'].fillna(credit_train['MonthlyIncome'].mean(),inplace=True)
credit_test['MonthlyIncome'].fillna(credit_test['MonthlyIncome'].mean(),inplace=True)
credit_train['NumberOfDependents'].fillna(credit_train['NumberOfDependents'].mode()[0], inplace=True)
credit_test['NumberOfDependents'].fillna(credit_test['NumberOfDependents'].mode()[0], inplace=True)
credit_test['MonthlyIncome'].fillna(credit_test['MonthlyIncome'].mean(),inplace=True)
plt.figure(figsize=(15,10))

sns.heatmap(credit_train.isnull(),yticklabels=False,cbar=False)
plt.figure(figsize=(15,10))

sns.heatmap(credit_train.corr(),annot=True)
credit_train
credit_test
print(credit_train.isnull().sum())
print(credit_test.isnull().sum())
X_train = credit_train[['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio', 'MonthlyIncome',

       'NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse',

       'NumberOfDependents']]

y_train = credit_train['SeriousDlqin2yrs']

X_test = credit_test[['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio', 'MonthlyIncome',

       'NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse',

       'NumberOfDependents']]
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

rfc_proba = rfc.predict_proba(X_test)
df=pd.DataFrame(rfc_proba,columns=['Id','Probability'])
df.head()
ind=credit_train['Unnamed: 0']

df['Id']=ind
df.head()
export_csv = df.to_csv('credit_score_random_forest.csv',index = None,header=True)