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
df.info()
df.head(7)
import seaborn as sns

sns.set()
sns.boxplot(df['Rating']);
df['Rating'].mean()
df['Rating'].median()
df.groupby('Category')['Rating'].mean().sort_values(ascending = False)[1:].plot(kind='bar');
df['Size'].value_counts()
def size_transform(s):

    if s[-1] == 'M':

        return float(s[:-1])*1024 # size in kB

    elif s[-1] == 'k':

        return float(s[:-1])

    else:

        return 0
df['Size in kB'] = df['Size'].apply(size_transform)
sns.boxplot(df['Size in kB']);
df['Content Rating'].value_counts()
df['Installs'].value_counts()
df[df['Installs'] == 'Free']
df.drop(df[df['Installs'] == 'Free'].index, axis = 0, inplace = True)
df['Installs'].value_counts()
def install_transform(s):

    s = s.replace(',', '')

    s = s.replace('+', '')

    if len(s) == 1:

        s = 2.5

    return float(s)
df['Installs as number'] = df['Installs'].apply(install_transform)

df.head()
df['Reviews'].astype(int)
sns.jointplot(x = 'Installs as number', y = 'Size in kB', data = df)
df['Reviews'] = df['Reviews'].astype(int)

df.groupby('Installs')['Reviews'].sum().sort_values()[:-7].plot(kind = 'bar');
numeric = ['Installs as number', 'Size in kB']

df[numeric].corr(method='spearman')
numeric = ['Installs as number', 'Reviews']

df[numeric].corr(method='spearman')
def ver_clear(col):

    for it in range(len(col)):

        col[it] = col[it].replace('.', '')

        for i in range(len(col[it])):

            if col[it][i] < '0' or col[it][i] > '9':

                col[it] = col[it].replace(col[it][i], '')



    for it in range(len(col)):

        if col[it] == '':

            col[it] = 10000000

            

        while len(col[it]) != 8:

            col[it] += '0'

        

    return col

df['intVersion'] = ver_clear(df['Current Ver'])