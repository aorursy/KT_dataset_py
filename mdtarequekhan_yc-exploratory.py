import pandas as pd

import numpy as np

import datetime 



# plotting libraries

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

%matplotlib inline 



plt.style.use('fivethirtyeight') #plot style used by fivethirtyeight

mpl.rcParams['figure.figsize'] = (12.0, 7.0)
df=pd.read_csv('../input/companies.csv')

df.head()
print ("The total number of companies funded by YC since 2005:", df.shape[0])
sns.countplot(df.year)

plt.title('# of companies funded per year')

plt.ylabel('Number')
sns.countplot(df.batch)
print ("The total number of areas YC invests in", len(df.vertical.unique()))
(df.vertical.unique())
df['vertical']=['others' if pd.isnull(x) else x for x in df['vertical']]
sns.countplot(df.vertical)

plt.title('Type of companies funded')

plt.ylabel('# of companies')
print ("B2B companies form" ,round((df['vertical']=='B2B').value_counts()[1]/float(len(df))*100),"% of YC portfolio")
sns.countplot(df['vertical'],hue=df['year'])