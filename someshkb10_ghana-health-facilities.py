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
file1 = pd.read_csv('../input/health-facilities-gh/health-facilities-gh.csv')

df1 = pd.DataFrame(file1)



print(df1.info())

print(df1.shape)

print(df1.head(10))
file2 = pd.read_csv('../input/health-facilities-gh/health-facility-tiers.csv')

df2 = pd.DataFrame(file2)



print(df2.info())

print(df2.shape)
import matplotlib.pyplot as plt

import seaborn as sns 



plt.figure(figsize=(12, 6))

ax = sns.countplot(df1['Region'])

plt.title('Distribution by Regions')

plt.xlabel('Regions',fontsize=14)

plt.xticks(fontsize=10)

plt.ylabel('Count',fontsize=16)

plt.yticks(fontsize=12)
plt.tight_layout(pad=0.95)

plt.figure(figsize=(16,6))

ax=sns.countplot(df1['Ownership'])

plt.title('Ownership',fontsize=22)

plt.xticks(fontsize=18,rotation='vertical')

plt.yticks(fontsize=18)

plt.ylabel('Count',fontsize=22)

plt.xlabel('')
group1 = df1.groupby('Type')



print(group1.size())



print(group1.size().sum())



group_total = df1.groupby(['Type','Region'])



print(group_total.size())



print(group1.size().sum())
print(df1.groupby('Ownership').size())

print(df1.groupby('Ownership').size().sum())
print(df1.groupby('Region').size())

 