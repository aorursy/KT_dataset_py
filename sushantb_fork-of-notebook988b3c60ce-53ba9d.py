%matplotlib inline

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

pd.options.display.max_rows = 1000

pd.options.display.max_columns = 20



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/train.csv')
train.head()
print (train.columns)
print (len(train.dtypes[train.dtypes==object]))

print (len(train.dtypes[train.dtypes!=object]))

print(len(train))
train['Id'].dtype

qualitative=[f for f in train.columns if train[f].dtype==object]

quantitative=[f for f in train.columns if train[f].dtype!=object]

quantitative.remove('SalePrice')

quantitative.remove('Id')

print (len(qualitative))

print (len(quantitative))

#Look for missing values now
print ("Hello")

missing=train.isnull().sum()

print (missing)

missing=missing[missing>0]

missing.sort_values(inplace=True)

missing.plot.bar()
import scipy.stats as st

y = train['SalePrice']
plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=st.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=st.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=st.lognorm)
normal= pd.DataFrame(train[quantitative])

normal=normal.apply(stats.normaltest)

print (normal)


