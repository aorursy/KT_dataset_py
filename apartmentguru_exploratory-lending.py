%matplotlib inline

import numpy as np

from scipy.stats import kendalltau

import seaborn as sns

import pandas as pd

df = pd.read_csv("../input/loan_data.csv", low_memory=False)

df.head()
df.info()
df.describe()
df['t'].value_counts()
df['loan_amnt'].hist(bins=10)
df['int_rate'].hist(bins=50)

# seems like most are concentrated between 10% - 15% interested rates
df.boxplot(column='int_rate')

# I suspected right
df.boxplot(column='int_rate', by = 'term')

#idk what this will do just curious I guess, will check if there is a sex or education level bias.
print (df.boxplot(column='loan_amnt'))
df.boxplot(column='loan_amnt', by='term')



df.apply(lambda x: sum(x.isnull()),axis=0)

# I feel good getting around to this shit.

## I need to find all of the missing values in the dataset
df[-20:]
df['grade'].unique
df['grade'].hist