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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
suicide = pd.read_csv('../input/master.csv', usecols = [0,1,2,3,4,5,11], index_col=['country', 'year']).sort_index()

suicide.head(10)
suicide.groupby('year').suicides_no.sum()
# Year wise Plot

plt.figure(figsize=(15,10))

suicide.groupby('year').suicides_no.sum().plot(kind = 'bar')

plt.xticks(fontsize=15, rotation=45)

plt.yticks(fontsize=15, rotation=0)

plt.show()
suicide.suicides_no.sum()
suicide.groupby('sex').suicides_no.sum()
suicide.groupby(['year','sex']).suicides_no.sum().head(10)
# Yearly plot of Male and Female Suicides

plt.figure(figsize=(15,10))

suicide.groupby(['year','sex']).suicides_no.sum().plot(kind = 'bar')



plt.show()
suicide.groupby(['country']).suicides_no.sum().head(10)
suicide.groupby(['country']).suicides_no.sum().tail(10)
suicide.groupby(['country']).suicides_no.sum().loc['United States']
suicide.groupby(['country']).suicides_no.sum().loc['Russian Federation']
# Sorted country wise suicide Plot

plt.figure(figsize=(15,10))

suicide.groupby(['country']).suicides_no.sum().sort_values().plot(kind = 'bar')
suicide.groupby(['country','age']).suicides_no.sum().head(10)
suicide.groupby(['country','age']).sum().loc['United States']
suicide.groupby(['age']).sum().head(12)
# Worldwide suicides by age group

plt.figure(figsize=(10,8))

suicide.groupby(['age']).suicides_no.sum().sort_values().plot(kind = 'bar')

plt.xticks(fontsize=10, rotation=45)

plt.yticks(fontsize=15, rotation=0)

plt.show()
# World wide male and female age group suicides 

plt.figure(figsize=(10,8))

suicide.groupby(['age', 'sex']).suicides_no.sum().sort_values().plot(kind = 'bar')



plt.show()
suicide.groupby(['country','sex']).suicides_no.sum().head(10)
plt.figure(figsize=(50,30))

suicide.groupby(['country', 'sex']).suicides_no.sum().plot(kind = 'bar')
suicide2 = pd.read_csv('../input/master.csv')

suicide2.head()
df = suicide2.pivot_table(values='suicides_no', index='country', columns='sex',aggfunc='sum')

df.head()
df.tail()
df['Total_f_m'] = df['female'] + df['male']

df.head()
import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(50,30))

df.groupby(['country']).Total_f_m.sum().plot(kind = 'bar')

plt.show()
df.loc['United States','female']
import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(50,30))

df.groupby(['country']).female.sum().sort_values().tail(20).plot(kind = 'bar')

plt.xticks(fontsize=25, rotation=0)

plt.yticks(fontsize=15, rotation=0)

plt.show()
import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(50,30))

df.groupby(['country']).male.sum().sort_values().tail(20).plot(kind = 'bar')

plt.xticks(fontsize=25, rotation=45)

plt.yticks(fontsize=25, rotation=0)

plt.show()
suicide.groupby(['sex','country']).suicides_no.sum()
suicide.groupby(['sex','country']).suicides_no.sum().loc[('female','United States')]
suicide.groupby(['sex','country']).suicides_no.sum().loc[('male','Japan')]