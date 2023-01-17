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
df1=pd.read_csv('../input/covid19-india-dataset/statewise.csv')

df2=pd.read_csv('../input/covid19-india-dataset/cases_time_series.csv')
df1.head()

cols1=['State','Confirmed','Recovered','Deaths','Active']

df1=df1[cols1]

df1.head()
import matplotlib.pyplot as plt

%matplotlib inline
plt.pie(df1['Confirmed'], labels=df1['State'])
plt.pie(df1['Deaths'], labels=df1['State'])
import seaborn as sns



sns.jointplot(x='Recovered', y='Deaths', data=df_death)
sns.barplot(x='State', y='Deaths', data=df1)
df_death=df1.loc[df1['Deaths']>=1.0]
plt.pie(df_death['Deaths'], labels=df_death['State'])
plt.pie(df_death['Confirmed'], labels=df_death['State'])
plt.pie(df_death['Recovered'], labels=df_death['State'])
plt.pie(df_death['Active'], labels=df_death['State'])
df2.head()
import re
def cln_date(txt):

    txt=re.sub('January', '.1',txt)

    txt=re.sub('February', '.2', txt)

    txt=re.sub('March', '.3', txt)

    txt=re.sub('[ ]', '', txt)

    return txt

    
df2['Date']=df2['Date'].apply(lambda x: cln_date(x))
df2.head()
plt.pie(df2['Daily Confirmed'], labels=df2['Date'])
sns.distplot(df2['Daily Confirmed'])
plt.subplot(2,2,1)



df2['Daily Confirmed'].plot.hist()

plt.subplot(2,2,2)

df2['Total Recovered'].plot.hist()
plt.bar(df2['Date'], df2['Daily Confirmed'])