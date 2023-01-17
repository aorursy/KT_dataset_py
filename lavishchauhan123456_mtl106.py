# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn import metrics

sns.set(style="ticks",context="talk")

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def plot(df):

    df=pd.DataFrame(df)

    for i in range(len(df.columns)):

        col=df.iloc[:,i]

        if(col.value_counts().count()>40 and (isinstance(col[0], np.float64)==True or isinstance(col[0],np.int64)==True)):

          col=col.dropna()

          if(sum(col==0)>0.75*len(col)):

             sns.distplot(col,kde_kws={'bw':1});

             plt.show()

          else:

             sns.distplot(col);

             plt.show()

    return 



df=pd.read_csv('/kaggle/input/births-in-us-1994-to-2003/births.csv')
df.head()
#very important ,see the task and the solution 

df.loc[(df.date_of_month==29) & (df.month==2)]
print(df.shape)

df.drop([789,2250],inplace=True)

df.shape
df.drop(['year','day_of_week'], axis = 1,inplace=True)

df.head()
d=df.groupby(['month','date_of_month'],as_index=False).sum()

d
d['births']=d['births']/d['births'].sum()
#d.drop(['month','date_of_birth'],axis=1,inplace=True)

pro=d['births']
plt.plot(pro)
pro=np.cumsum(pro)
pro
from bisect import bisect_left 

import random

def bs(a, x): 

    i = bisect_left(a, x) 

    if i: 

        return (i-1) 

    else: 

        return -1

s=set()

times = 10000

count=0

for i in range(times):

    for j in range(23):

        val=bs(pro,random.random())

        #print(val)

        if(val in s):

            count+=1

            s.clear()

            break

        s.add(val)

count
