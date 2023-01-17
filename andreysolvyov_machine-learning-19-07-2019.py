



import numpy as np 

import pandas as pd

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/googleplaystore.csv')

df.head(10)
df.info()
import seaborn as sns

sns.set()

sns.boxplot(df['Rating']);
print(df['Rating'].median())

print(df['Rating'].mean())
df.groupby('Category')['Rating'].mean().sort_values()[::-1][1:].plot(kind='bar')
def si(s):

    if s[-1] == 'M':

        return float(s[:-1])*1024

    elif s[-1] == 'k':

        return float(s[:-1])

    else:

        return 0
df['Size in kb'] = df['Size'].apply(si)