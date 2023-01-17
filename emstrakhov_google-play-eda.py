# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/googleplaystore.csv')

df.head(10)
df.info()
import seaborn as sns

sns.set()
sns.boxplot(df['Rating']);
df['Rating'].median()
df['Rating'].mean()
df.groupby('Category')['Rating'].mean().sort_values(ascending=False)[1:].plot(kind='bar')
df['Size'].value_counts()
def size_transform(s):

    if s[-1] == 'M':

        return float(s[:-1]) * 1024 # size in KB

    elif s[-1] == 'k':

        return float(s[:-1])

    else:

        return 0
df['Size in KB'] = df['Size'].apply(size_transform)
df['Size in KB']
sns.boxplot(df['Size in KB']);
df['Content Rating'].value_counts()
df['Installs'].value_counts()
df[df['Installs'] == 'Free']
df.drop(df[df['Installs'] == 'Free'].index, axis=0, inplace=True)
df['Installs'].value_counts()
def installs_transform(s):

    s = s.replace(',', '')

    s = s.replace('+', '')

    if len(s) == 1:

        s = 2.5

    return float(s)
df['Installs as number'] = df['Installs'].apply(installs_transform)

df.head()
df['Reviews'].dtype
df['Reviews'] = df['Reviews'].astype(int)
import seaborn as sns

sns.jointplot(x='Reviews', y='Installs as number', data=df);
df.groupby('Installs')['Reviews'].sum().sort_values()[:-7].plot(kind='bar');