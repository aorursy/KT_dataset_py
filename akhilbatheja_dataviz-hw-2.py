# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/lending-club-loan-data/loan.csv', low_memory=False)

data.drop(['id', 'member_id', 'emp_title'], axis=1, inplace=True)



data.replace('n/a', np.nan,inplace=True)

data.emp_length.fillna(value=0,inplace=True)



data['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)

data['emp_length'] = data['emp_length'].astype(int)



data['term'] = data['term'].apply(lambda x: x.lstrip())
data.head()
data.columns
grade_order = data.grade.unique()

grade_order.sort()

subgrade_order = data.sub_grade.unique()

subgrade_order.sort()
sns.violinplot(y = "int_rate", x = "grade", data = data, order =grade_order )
sns.boxplot(y = "int_rate", x = "grade", data = data, order =grade_order )
plt.figure(figsize=(20,10))

sns.violinplot(y = "int_rate", x = "sub_grade", data = data, order = subgrade_order )
investment_grade_ratings = []

subinvestment_grade_ratings = []

for rating in subgrade_order:

  if rating <= "B3":

    investment_grade_ratings.append(rating)

  else:

    subinvestment_grade_ratings.append(rating)

print (investment_grade_ratings)

print (subinvestment_grade_ratings)
investment_grade_data = data[data.sub_grade.isin(investment_grade_ratings)]

subinvestment_grade_data = data[data.sub_grade.isin(subinvestment_grade_ratings)]
investment_grade_data.head()
investment_subgrade_order = investment_grade_data.sub_grade.unique()

investment_subgrade_order.sort()

sns.boxplot(y = "int_rate", x = "sub_grade", data = investment_grade_data, order =investment_subgrade_order )
subinvestment_subgrade_order = subinvestment_grade_data.sub_grade.unique()

subinvestment_subgrade_order.sort()

plt.figure(figsize=(20,10))

sns.boxplot(y = "int_rate", x = "sub_grade", data = subinvestment_grade_data, order =subinvestment_subgrade_order )
subinvestment_grade_data.term.unique()
investment_grade_data.term.unique()
sns.distplot(investment_grade_data.loan_amnt, rug = True, label = "Investment Grade")

sns.distplot(subinvestment_grade_data.loan_amnt, rug = True, label = "Subinvestment Grade")

plt.legend()
invgrade_shortduration = investment_grade_data[investment_grade_data.term=="36 months"]

invgrade_longduration = investment_grade_data[investment_grade_data.term == "60 months"]
sns.boxplot(y = "int_rate", x = "sub_grade", data = invgrade_shortduration, order =investment_subgrade_order )
sns.boxplot(y = "int_rate", x = "sub_grade", data = invgrade_longduration, order =investment_subgrade_order )
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)

sns.boxplot(y = "int_rate", x = "sub_grade", data = invgrade_shortduration, order =investment_subgrade_order )

plt.xlabel("Credit Rating")

plt.ylabel("Interest Rate")

plt.title("Short Duration Investment Grade Loans")





plt.subplot(1,2,2)

sns.boxplot(y = "int_rate", x = "sub_grade", data = invgrade_longduration, order =investment_subgrade_order )

plt.xlabel("Credit Rating")

plt.ylabel("Interest Rate")

plt.title("Long Duration Investment Grade Loans")


