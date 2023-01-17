# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# reading data from dataset

df = pd.read_csv("../input/mock_bank_data_original_PART1.csv")

df.head(10)
#  look at the current state of missing values in our dataset

df.isnull().sum()



#  checking account balance ("cheq_balance")   and savings account balances ("savings_balance") 

#  have 23 and 91 missing values, respectively.
# Let's compute the mean of cheq_balance by state



df.groupby(['state']).mean()['cheq_balance']
df.groupby(['state']).mean()['savings_balance']

# Let's go ahead and fill in these missing values by using the Pandas 'groupby' and 'transform' functionality, along with a lambda function. 

# We then round the result in the line of code beneath.
# Replace cheq_balance NaN with mean cheq_balance of same state

df['cheq_balance'] = df.groupby('state').cheq_balance.transform(lambda x: x.fillna(x.mean()))

df.cheq_balance = df.cheq_balance.round(2)

df.cheq_balance.head()
# Replace savings_balance NaN with mean savings_balance of same state

df['savings_balance'] = df.groupby('state').savings_balance.transform(lambda x: x.fillna(x.mean()))

df.savings_balance = df.savings_balance.round(2)

df.savings_balance.head()
# Checking the results 

df.isnull().sum()