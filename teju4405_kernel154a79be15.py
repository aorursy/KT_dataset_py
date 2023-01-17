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
df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')



df.head()



df.info()
df.fillna(method='bfill')
df.describe()


s=df.iloc[3:10,2:10]

s
df.hist()
s=pd.get_dummies(df['has_full_text'])

s
x=df['has_full_text'].value_counts()

x
from matplotlib import pyplot as plt

plt.plot(df['pubmed_id'])

plt.show()
plt.plot(df['Microsoft Academic Paper ID'])

plt.show()
df.corr()
import seaborn as sns

m=sns.regplot(df['pubmed_id'],df['Microsoft Academic Paper ID'])

plt.ylim(0,)
df.mean()
df.sort_values(by='sha')
bin=np.linspace(min(df['Microsoft Academic Paper ID']),max(df['Microsoft Academic Paper ID']),4)

y=['low','high','medium']

s=pd.cut(df['Microsoft Academic Paper ID'],bin,labels=y,include_lowest=True)

s
sns.distplot(df['pubmed_id'],hist=False,color='r',label='xyz')

plt.show()