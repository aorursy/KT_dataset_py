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

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import show

df =  pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
df.head()
df.info()
plt.figure(figsize=(30,10))

sns.heatmap(df.corr(),annot=True,linewidth=1)
plt.figure(figsize=(30,10))

ax = sns.barplot(x= 'country',y = 'exch_usd',data=df)

for p in ax.patches:

        ax.annotate('%{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))
plt.figure(figsize=(30,10))

ax = sns.barplot(x= 'country',y ='inflation_annual_cpi',data=df)

for p in ax.patches:

        ax.annotate('%{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))
plt.figure(figsize = (30,10))

ax = sns.countplot(x='country',hue='banking_crisis',data=df) 

for p in ax.patches:

        ax.annotate('%{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))
plt.figure(figsize = (30,10))

ax = sns.countplot(x='country',hue='systemic_crisis',data=df)

for p in ax.patches:

        ax.annotate('%{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))
plt.figure(figsize = (30,10))

ax = sns.countplot(x='country',hue='currency_crises',data=df)

for p in ax.patches:

        ax.annotate('%{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))
plt.figure(figsize = (30,10))

ax = sns.countplot(x='country',hue='inflation_crises',data=df)

for p in ax.patches:

        ax.annotate('%{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))
plt.figure(figsize = (30,10))

ax = sns.countplot(x='country',hue='independence',data=df)

for p in ax.patches:

        ax.annotate('%{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))
plt.figure(figsize = (30,10))

ax = sns.countplot(x='country',hue='sovereign_external_debt_default',data=df)

for p in ax.patches:

        ax.annotate('%{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))
plt.figure(figsize = (30,10))

ax = sns.countplot(x='country',hue='domestic_debt_in_default',data=df)

for p in ax.patches:

        ax.annotate('%{:.1f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+1))
sns.pairplot(df)