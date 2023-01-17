# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

df=pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

df.set_index("Name",inplace=True)
import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(8,6))

h=pd.value_counts(df['Genre'],sort=False)

sns.barplot(y=h.index,x=h)
u=pd.value_counts(df['Platform'])

plt.figure(figsize=(10,8))

sns.barplot(y=u.index,x=u)
i=pd.value_counts(df['Year'],sort=False)

s=df.groupby("Year")['Global_Sales'].agg('sum')
year_sales=pd.merge(i,s,left_index=True,right_index=True)
year_sales.columns=['Number of games','Global_Sales']

year_sales.sort_index(inplace=True)
plt.figure(figsize=(20,6))

sns.lineplot(x=year_sales.index,y=year_sales['Global_Sales'])

plt.show()

plt.figure(figsize=(20,6))

sns.barplot(x=year_sales.index,y=year_sales['Number of games'])
j=df.groupby('Platform')['Global_Sales'].agg('sum')

plt.figure(figsize=(20,6))

sns.barplot(x=j.index,y=j)
j=df.groupby('Genre')['Global_Sales'].agg('sum')

plt.figure(figsize=(15,6))

sns.barplot(x=j.index,y=j)
o=df.groupby(['Platform','Genre'])['Global_Sales'].agg("sum")

d=np.max(o.unstack(level=1),axis=0)
num=0

for genre in d.index:

    g=o.unstack(level=1)[o.unstack(level=1)[genre]==d[num]]

    print(g.index," ",genre)

    num+=1
num=0

d=np.max(o.unstack(level=0),axis=0)

for genre in d.index:

    g=o.unstack(level=0)[o.unstack(level=0)[genre]==d[num]]

    print(g.index," ",genre)

    num+=1
h=df.groupby(['Publisher','Platform'])['Rank'].agg('count')
plt.figure(figsize=(15,10))

p=h.unstack(level=0).replace({np.nan:0})

sns.heatmap(p,cmap='tab20')

p
pub=df.groupby("Publisher")['NA_Sales','JP_Sales','EU_Sales','Other_Sales','Global_Sales'].agg('sum')
print('Publisher with max global sales',pub[pub['Global_Sales']==max(pub['Global_Sales'])].index)

print('Publisher with max sales in North America ',pub[pub['NA_Sales']==max(pub['NA_Sales'])].index)

print('Publisher with max sales in Japan ',pub[pub['JP_Sales']==max(pub['JP_Sales'])].index)

print('Publisher with max sales in the rest of the world ',pub[pub['Other_Sales']==max(pub['Other_Sales'])].index)

print('Publisher with max sales in Europe ',pub[pub['EU_Sales']==max(pub['EU_Sales'])].index)
sns.heatmap(df.groupby('Genre')['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].agg('sum'),annot=True)
plt.figure(figsize=(8,8))

sns.heatmap(df.groupby('Platform')['NA_Sales','EU_Sales','JP_Sales','Other_Sales'].agg('sum'))
sns.lineplot(x=df['Rank'],y=df['Global_Sales'])