# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #  plotting tool

import seaborn as sns  #graphing tool 

import math





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/oneYear.csv', encoding='latin-1')

df.head()
pd.isnull(df).any()   


ax = df.plot(kind = 'bar')
df.groupby(['Economy']).head()

df.drop_duplicates('Economy', keep=False)
import matplotlib.ticker as ticker

tick_spacing = 5

plt.figure(figsize=(15,15))

x = df['Ease of doing business rank global (DB19)']

y = df['Score-Trading across borders(DB16-19 methodology)']

plt.scatter(df['Ease of doing business rank global (DB19)'], df['Score-Trading across borders(DB16-19 methodology)'])

plt.xlabel('Rank')

plt.ylabel('Score')

plt.xticks(min(x), max(x))

plt.show()
fig1, ax1 = plt.subplots()

ax1.set_title('Breakdown of how easy it is to do business globally')

ax1.boxplot(df['Ease of doing business score global (DB17-19 methodology)'])
plt.hist(df['Ease of doing business score global (DB17-19 methodology)'])
def logFunction(x):

  return np.log(x)



df1 = df['Ease of doing business score global (DB17-19 methodology)'].apply(logFunction)

plt.hist(np.log(df1))
df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
df.drop(df.loc[df['Trading across Borders - Time to export: Documentary compliance (hours) (DB16-19 methodology)']=='No Practice'].index, inplace=True)

df.drop(df.loc[df['Rank-Trading across borders (DB19)']== ' '].index, inplace=True)
df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
temp = df['Ease of doing business score global (DB17-19 methodology)'].idxmin()

df.drop(df.loc[df['Economy']== 'Tonga'].index, inplace=True)

df.drop(df.loc[df['Economy']== 'Trinidad and Tobago'].index, inplace=True)
fig1, ax1 = plt.subplots()

ax1.set_title('Basic Plot')

ax1.boxplot(df['Ease of doing business score global (DB17-19 methodology)'], showfliers = False)
fig1, ax1 = plt.subplots()

ax1.set_title('Basic Plot')

ax1.boxplot(df['Trading across Borders - Time to export: Documentary compliance (hours) (DB16-19 methodology)'].astype(np.float))
plt.hist(df['Trading across Borders - Time to export: Documentary compliance (hours) (DB16-19 methodology)'].astype(np.float))
df.describe()
df[pd.to_numeric(df['Ease of doing business rank global (DB19)'], errors = 'coerce').notnull()]

corr = df['Score-Trading across borders(DB16-19 methodology)'].corr(df['Ease of doing business score global (DB17-19 methodology)'])

print(corr)