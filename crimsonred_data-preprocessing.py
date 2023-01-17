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
data = pd.read_csv("/kaggle/input/lendingdata.csv")

data.shape
count = 0

for i in data.dtypes:

  if(i==object):

    count+=1

print(count)
sum(data.isnull().sum())
data['borrower_genders'].fillna(data['borrower_genders'].mode()[0], inplace=True)

data['borrower_genders'].isna().sum()
bool = data['activity'].duplicated()

print(len(data[~bool]))

bool = data['country_code'].duplicated()

print(len(data[~bool]))

bool = data['distribution_model'].duplicated()

print(len(data[~bool]))

bool = data['status'].duplicated()

print(len(data[~bool]))
data['term_in_months']= data['term_in_months'].replace('1 Year',12)

data['term_in_months']= data['term_in_months'].replace('2 Years',24)

data.dtypes
data.loan_amount.quantile([0.75])
df = data[data['sector']=='Agriculture']

min(df['loan_amount'])
churn = pd.read_csv('/kaggle/input/churn.csv')

churn.dtypes
churn['customerID'].duplicated().sum()
churn['MonthlyCharges'].mean()
sum(churn['Dependents']=='1@#')
churn['tenure'].dtypes
churn["tenure"]= churn["tenure"].replace("One",1) 

churn["tenure"]= churn["tenure"].replace("Four",4)
churn.head()