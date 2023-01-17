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
df.head()
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