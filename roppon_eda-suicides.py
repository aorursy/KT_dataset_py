# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.



file = '../input/master.csv'



df = pd.read_csv(file)



print(df.shape)
df.head()
df[['country', 'year']].nunique()
df.groupby('sex').agg({'suicides_no': 'sum'})
df.groupby(['sex', 'generation']).agg({'suicides_no': 'sum'})
d = df.groupby(['sex', 'year']).agg({'suicides/100k pop': 'mean'})



d.head()
d.reset_index().head()
plt.style.use('ggplot')
plt.figure()

df.hist(figsize=(8, 6), layout=(2, 3))

plt.tight_layout();
d = df.groupby(['sex', 'year']).agg({'suicides/100k pop': 'mean'})

d.reset_index(inplace=True)

female = d.loc[d['sex']=='female']

male = d.loc[d['sex']=='male']

f, ax = plt.subplots(ncols=2, figsize=(12, 5))

female.plot(kind='bar', x='year', y='suicides/100k pop', label='female', ax=ax[0], alpha=0.5)

male.plot(kind='bar', x='year', y='suicides/100k pop', label='male', ax=ax[1], alpha=0.5)

f.tight_layout(rect=[0, 0.02, 1, 0.95])

f.suptitle('Suicides/100k Population')
f, ax = plt.subplots()

df.boxplot(by='sex', column=['suicides/100k pop'], ax=ax)

f.tight_layout()

f.suptitle(None)
d = df.groupby('country').agg({'suicides/100k pop': 'mean'}).sort_values(by='suicides/100k pop')

f, ax = plt.subplots(figsize=(8, 14))

d.plot(kind='barh', ax=ax)
d = df[df['country']=='Thailand'].groupby('year').agg({'suicides/100k pop': 'mean'})

f, ax = plt.subplots(figsize=(8, 8))

d.plot(kind='barh', ax=ax)