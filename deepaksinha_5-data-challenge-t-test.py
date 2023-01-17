import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



import scipy.stats as stats



import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline



# Read data in pandas data frame

df = pd.read_csv('../input/museums.csv', low_memory=False)

df.head()
df.dtypes
df['Museum Type'].value_counts()
# Add variable Type to distinguish Zoo and Others

df['Type'] = df['Museum Type'].apply(lambda x: 'ZOO' if re.search('ZOO', x) else 'OTHERS')
df['Type'].value_counts()
df.groupby(['Museum Type', 'Type']).size()
df['Revenue'].isnull().sum()
# mean impute

df['Revenue'].fillna(df['Revenue'].mean(), inplace=True)
df['Revenue'].isnull().sum()
df[df['Revenue'] == 0]['Revenue'].count()
# Create data point 

x = df[df['Type'] == 'ZOO']['Revenue']

y = df[df['Type'] == 'OTHERS']['Revenue']
x = x[x>0]

y = y[y>0]
# Quantile-Quantile Plot using SciPy (QQ)

# qq plot is used to check whether the data is distributed normally or not.



plt.subplot(221)

stats.probplot(x, dist="norm", plot=plt)

plt.subplot(222)

stats.probplot(y, dist="norm", plot=plt)



plt.show()
x = (x+1).apply(np.log)

y = (y+1).apply(np.log)
# QQ plot with logrithmic data

plt.subplot(221)

stats.probplot(x, dist="norm", plot=plt)

plt.subplot(222)

stats.probplot(y, dist="norm", plot=plt)



plt.show()
plt.figure(figsize = (15,7))

sns.distplot(x)

sns.distplot(y)
zoo = df[df['Type'] == 'ZOO']['Revenue']

others = df[df['Type'] == 'OTHERS']['Revenue']



zoo = zoo[zoo > 0]

others = others[others > 0]



zoo = (zoo+1).apply(np.log)

others = (others+1).apply(np.log)

print('Mean for Zoo:  {}'.format(zoo.mean()))

print('Mean for others:  {}'.format(others.mean()))
# T-test to check whether the revenue for zoos is different than all other types of museums 



# Two-Sample T-Test



stats.ttest_ind(a= zoo,

                b= others,

                equal_var=False) 
# Data Population

revpop = df[df['Revenue'] > 0 ]['Revenue']
revpop = (revpop+1).apply(np.log)
print('Mean for Zoo:  {}'.format(zoo.mean()))

print('Mean for All Population:  {}'.format(revpop.mean()))
stats.ttest_1samp(a= zoo,               # Sample data

                 popmean= revpop.mean())  # Pop mean