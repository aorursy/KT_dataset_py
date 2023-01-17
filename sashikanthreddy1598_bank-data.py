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

import missingno as msno
bank_data = pd.read_csv("../input/bank.csv",sep=";")

bank_data.head()
bank_data.tail()
msno.matrix(bank_data)
#lets find out numerical features in the Bank_data

numeric_features = bank_data.select_dtypes(include=[np.number])



numeric_features.columns
categorical_features = bank_data.select_dtypes(include=[np.object])



categorical_features.columns
x = bank_data.job.value_counts()

print(x.idxmax()) #solution 1
print(x[x == x.max()]) 
print(x[x == x.max()].index[0]) #solution 3
bank_data.marital.value_counts()
bank_data.education.value_counts()
# above checks are to see if the required levels are represented in any other format

bank_data.loc[(bank_data.marital == 'married') & (bank_data.education == 'tertiary'),'balance'].mean() #solution1
#bank balance for marital status as 'single' & the customers who already finished their 'tertiary ' education

bank_data.loc[(bank_data.marital == 'single') & (bank_data.education == 'tertiary'), 'balance'].mean() #solution 2
#Bank balance for those whose marital status as ' divorced' & those people who already finished their 'tertiary' education

bank_data.loc[(bank_data.marital == "divorced") & (bank_data.education == 'tertiary'), 'balance'].mean() #solution 3
#Bank balane for marital status as 'married' & the customer who have done only 'primary' education



bank_data.loc[(bank_data.marital == 'married') & (bank_data.education == 'primary'),'balance'].mean() #solution
#Bank balance for those are divorsed & the customer done only 'primary' education

bank_data.loc[(bank_data.marital == 'divorced') & (bank_data.education == 'primary'),'balance'].mean() #solution
bank_data.groupby('education').agg({'age':'mean'})
bank_data.drop(['job', 'day', 'month', 'pdays', 'previous', 'poutcome'],axis=1, inplace=True)

bank_data.head()
cat_cols = bank_data.columns[bank_data.dtypes == 'object']

num_cols = bank_data.columns[bank_data.dtypes == 'object']



two_level_cols = []

more_level_cols = []



for col in cat_cols:

    if len(pd.unique(bank_data[col])) ==2:

        two_level_cols.extend([col])

    else:

        more_level_cols.extend([col])

print(two_level_cols)

print(more_level_cols)
for col in two_level_cols:

    uq_values = pd.unique(bank_data[col])

    print(uq_values)

    bank_data.loc[bank_data[col]==uq_values[0], col] = 0

    bank_data.loc[bank_data[col]==uq_values[1], col] = 1

    print(pd.unique(bank_data[col]))
bank_data = pd.get_dummies(bank_data, columns=more_level_cols)

bank_data.head()