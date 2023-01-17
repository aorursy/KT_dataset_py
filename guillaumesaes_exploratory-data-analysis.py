import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from scipy.stats import shapiro

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
house = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

pd.set_option('display.max_row',111)

pd.set_option('display.max_column',111)

house.head()
df = house.copy()

n, m = df.shape

target = 'SalePrice'

target_min = np.min(df[target])

target_max = np.max(df[target])





print('The number of data: ', n)

print('The number of variable: ', m)



print('The target is: ', target)

print('The type of target: ', np.dtype(df[target]))

print(f'The value of the targets vary from {target_min} to {target_max}')

print('Here are all types of variables:')

print(df.dtypes.value_counts())



plt.figure(figsize=(12,4))

sns.set()

sns.countplot(df[target])

plt.show()
plt.figure(figsize=(12,8))

sns.heatmap(df.isna(), cbar=False)

plt.show()



print('Les proportions de données manquantes :')

print((df.isna().sum()/n).sort_values())
df = df[df.columns[(df.isna().sum()/n) < 0.4]]

df = df.drop('Id', axis = 1)

df.head()
plt.figure(figsize=(12,8))

sns.heatmap(df.isna(), cbar=False)

plt.show()



print('Les proportions de données manquantes :')

print((df.isna().sum()/n).sort_values())
n, m = df.shape

target = 'SalePrice'

target_min = np.min(df[target])

target_max = np.max(df[target])





print('The number of data: ', n)

print('The number of variable: ', m)



print('The target is: ', target)

print('The type of target is: ', np.dtype(df[target]))

print(f'The value of the targets vary from {target_min} to {target_max}')

print('Here are all types of variables:')

print(df.dtypes.value_counts())



plt.figure(figsize=(12,4))

sns.set()

sns.distplot(df[target])

plt.show()
np.sum(df['SalePrice']>450000)
plt.figure(figsize=(12,4))

sns.set()

sns.distplot(df[target])

plt.show()



def normal_test(col):

    alpha = 0.02

    stat, p = shapiro(col)

    if p < alpha:

        return 'H0 rejetée'

    else:

        return 0

    

normal_test(df[target])
p = int(np.sqrt(df.columns[df.dtypes == 'float64'].size))+1

plt.figure(figsize=(20,8))

i = 0

for var in df.columns[df.dtypes == 'float64']:

    plt.subplot(p,p,i+1)

    sns.distplot(df[var])

    i += 1

plt.tight_layout()

plt.show()
p = int(np.sqrt(df.columns[df.dtypes == 'int64'].size))+1

plt.figure(figsize=(20,20))

i = 0

for var in df.columns[df.dtypes == 'int64']:

    plt.subplot(p, p, i+1)

    sns.distplot(df[var], kde_kws={'bw': 0.1})

    i += 1

plt.tight_layout()

plt.show()
for col in df.select_dtypes('object'):

    print(f'For the variable {col :-<20} we have the elements {df[col].unique()}')
p = int(np.sqrt(df.columns[df.dtypes == 'object'].size))+1

plt.figure(figsize=(20,18))

i = 0

for var in df.columns[df.dtypes == 'object']:

    plt.subplot(p, p, i+1)

    df[var].value_counts().plot.pie()

    i += 1

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,12))

sns.pairplot(df[df.columns[df.dtypes == 'float64']])

plt.show()
df.corr()[target].sort_values()
plt.figure(figsize=(8,8))

sns.pairplot(df[[target,'OverallQual']])

plt.show()
plt.figure(figsize=(8,8))

sns.pairplot(df[[target,'GrLivArea']])

plt.show()
df.corr()
plt.figure(figsize=(12,8))

sns.heatmap(df.corr())

sns.clustermap(df.corr())

plt.show()