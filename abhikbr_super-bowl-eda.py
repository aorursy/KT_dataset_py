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
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

%matplotlib inline



from datetime import datetime
df=pd.read_csv('../input/superbowl-history-1967-2020/superbowl.csv')

df.head()
df.info()
df.describe()
df.isnull().sum()
df['Date']=df['Date'].apply(lambda x: datetime.strptime(x, '%b %d %Y').strftime('%d/%m/%Y'))

df.head()
plt.figure(figsize=(25,12))

c=sns.countplot(df['Winner'],palette="rainbow")

c.set_xticklabels(c.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.figure(figsize=(25,12))

c=sns.countplot(df['Loser'],palette="Set1")

c.set_xticklabels(c.get_xticklabels(), rotation=45, horizontalalignment='right')
l=df['Loser'].unique().tolist()

l
only_win=df[(~df['Winner'].isin(l))]

only_win
plt.figure(figsize=(12,6))

sns.lineplot(x=only_win['Winner'],y=only_win['Winner Pts'],palette="Set1")

plt.title("Only Winners")
w=df['Winner'].unique().tolist()

only_lose=df[(~df['Loser'].isin(w))]

plt.figure(figsize=(20,8))

sns.lineplot(x=only_lose['Loser'],y=only_lose['Loser Pts'],palette="Set1")

plt.title("Only Loser")
b=only_lose['Loser'].tolist()
b
w_pts=df['Winner Pts'].sort_values(ascending=False).tolist()

w_pts[:5]
top_5=df[df['Winner Pts'].isin(w_pts[:5])]

top_5=top_5.sort_values(by='Winner Pts')

plt.figure(figsize=(12,6))

sns.barplot(x='Winner',y='Winner Pts',data=top_5)
top_5
l_pts=df['Winner Pts'].sort_values(ascending=False).tolist()

l_pts[-5:]
last_5=df[df['Loser Pts'].isin(l_pts[-5:])]

last_5=last_5.sort_values(by='Loser Pts')

plt.figure(figsize=(15,10))

sns.barplot(x='Loser',y='Loser Pts',data=last_5)
df['Match']=df[['Winner', 'Loser']].apply(lambda x: ' vs '.join(x), axis=1)

df.head()
plt.figure(figsize=(25,20))

sns.countplot(y=df['MVP'])
a=df['State'].value_counts().reset_index()

a=a.rename(columns={'index':'State','State':'counts'})

a
fig = plt.figure(figsize=(12,6))

ax = fig.add_axes([0,0,1,1])

ax.axis('equal')

ax.pie(a['counts'], labels = a['State'],autopct='%1.2f%%')

plt.show()
cm=sns.light_palette("green",as_cmap=True)

table=pd.pivot_table(df,index=['State','City','Stadium','Date','Winner','Loser'])

s=table.style.background_gradient(cmap=cm)

s