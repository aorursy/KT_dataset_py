# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/instant-gratification/train.csv')
df.head()
(df.isna().sum() !=0 ).sum()
df0 = df[df['target'] == 0].drop(['id', 'target'], axis=1).reset_index(drop=True)

df1 = df[df['target'] == 1].drop(['id', 'target'], axis=1).reset_index(drop=True)
df.describe()
df0.describe()
df1.describe()
dfcor = df.corr(method ='pearson')

dfcor
dfcor0 = df0.corr(method ='pearson')

dfcor0.head()
dfcor1 =  df1.corr(method ='pearson')

dfcor1.head()
plt.figure(figsize=(20, 15))

plt.plot( df.drop(['id', 'target'], axis=1) )

plt.show()
plt.figure(figsize=(20, 15))

plt.plot(df0 )

plt.show()
plt.figure(figsize=(20, 15))

plt.plot(df1)

plt.show()
temp = df.drop(['id', 'target'], axis=1)

plt.figure(figsize=(20, 15))

plt.plot( temp.loc[1] )

plt.plot( temp.loc[1000] )

plt.show()
plt.figure(figsize=(20, 15))

plt.plot( df0.loc[1] )

plt.plot( df0.loc[1000] )

plt.show()


plt.figure(figsize=(20, 15))

plt.plot( df1.loc[1] )

plt.plot( df1.loc[1000] )

plt.show()
dfcor[['target']].sort_values(by='target')
ax = sns.heatmap(dfcor*100)
scor = abs((dfcor - dfcor0) - (dfcor - dfcor1))

scor
(scor > 0.01).sum().sum()
plt.figure(figsize=(20, 20))

sns.heatmap(scor, cbar=False)
plt.figure(figsize=(20, 50))

sns.heatmap(scor[:100], cbar=False)
plt.figure(figsize=(20, 50))

sns.heatmap(scor[101:200], cbar=False)
plt.figure(figsize=(20, 50))

sns.heatmap(scor[201:], cbar=False)
dfcor_n = pd.DataFrame(

                  MinMaxScaler().fit_transform(scor), 

                  index = scor.index, 

                  columns=scor.columns)



dfcor_n.head()
plt.figure(figsize=(20, 50))

sns.heatmap(dfcor_n, cbar=False)
plt.figure(figsize=(20, 50))

sns.heatmap(dfcor_n[:100], cbar=False)
plt.figure(figsize=(20, 50))

sns.heatmap(dfcor_n[101:200], cbar=False)
plt.figure(figsize=(20, 50))

sns.heatmap(dfcor_n[201:], cbar=False)
dfcor_t = pd.DataFrame(

                  MinMaxScaler().fit_transform(dfcor), 

                  index = dfcor.index, 

                  columns=dfcor.columns)



dfcor_t.head()
plt.figure(figsize=(20, 50))

sns.heatmap(dfcor_t, cbar=False)
dfr = df.drop(['id', 'wheezy-copper-turtle-magic', 'target'], axis=1)
dfr0 = df[df['target'] == 0].drop(['id', 'target', 'wheezy-copper-turtle-magic'], axis=1).reset_index(drop=True)

dfr1 = df[df['target'] == 1].drop(['id', 'target', 'wheezy-copper-turtle-magic'], axis=1).reset_index(drop=True)
corfr= dfr.corr(method ='pearson')

corfr
corfr0= dfr0.corr(method ='pearson')

corfr0
corfr1= dfr1.corr(method ='pearson')

corfr1
scor2 = abs((corfr - corfr0) - (corfr - corfr1) )

scor2
dfcor_nt = pd.DataFrame(

                  MinMaxScaler().fit_transform(scor2), 

                  index = scor2.index, 

                  columns=scor2.columns)



dfcor_nt.head()
plt.figure(figsize=(25, 25))

sns.heatmap(dfcor_nt, cbar=False)