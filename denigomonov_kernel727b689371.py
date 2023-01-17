import numpy as np

import pandas as pd

import seaborn as sns
df=pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')

df.head()
df.describe()
len(df)
df.drop('date', axis=1, inplace=True)

df.head()
sns.set(rc={'figure.figsize':(10,4)})

sns.boxplot(x='month', y='number', data=df, color='red')
sns.set(rc={'figure.figsize':(16,4)})

sns.boxplot(x='state', y='number', data=df, color='red')