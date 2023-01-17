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
import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
df.head(50)
df.isnull().sum()
df.columns
df.rename(columns={'State/UnionTerritory':'State'},inplace=True)
df.columns
df['Confirmed'].plot.hist()
sns.countplot(x='Deaths',data=df,hue='State',palette='mako')
df['Confirmed'].describe()
top=df.nlargest(20,'Confirmed')
sns.stripplot(x='State',y='Confirmed',data=top)
fig,ax = plt.subplots(1)

fig.set_size_inches(14,6)

sns.barplot(df["State"],df["Confirmed"])

plt.xticks(rotation=45,fontsize=10)

plt.title("Confirmed Cases Statewise as on ",fontsize=16)

plt.xlabel("State/Union Territory",fontsize=14)

plt.ylabel("Confirmed Cases",fontsize=14)

plt.show()
from matplotlib import style

style.use('ggplot')



df.plot(x='Date',y='Confirmed',kind='line',linewidth=5,color='r',figsize=(25,15))

plt.ylabel('Corona Cases')



plt.grid()

plt.show()
sns.relplot(x='Confirmed',y='Deaths',hue = 'State',data=df,height=8)