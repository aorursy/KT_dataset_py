# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
import pandas as pd

data=pd.read_csv("../input/winemag-data_first150k.csv")

data.head(10)
data['country'].value_counts().head(10).plot.bar()
data['province'].value_counts().head(10).plot.bar()
data['points'].value_counts().sort_index().plot.bar()
data['points'].value_counts().sort_index().plot.area()
print(data['price'].mean())

print(data['price'].std())
data[data['price']<500]['price'].plot.hist()
data[data['price']>1000]
data[data['price']<80].sample(70).plot.scatter(x='price',y='points')
data[data['price']<100].plot.hexbin(x='price',y='points',gridsize=10)
data.head()
import seaborn as sns
sns.countplot(data['points'])
sns.kdeplot(data.query('price<100').price

           )
sns.kdeplot(data[data['price']<100].loc[:,['price','points']].dropna().sample(2500))
sns.distplot(data['points'],kde=False,bins=10)
sns.jointplot(x='price',y='points',data=data[data['price']<100])
sns.jointplot(x='price',y='points',data=data[data['price']<100],kind='hex',gridsize=20)
data[data.variety.isin(data.variety.value_counts().head(5).index)]



d1=sns.boxplot(x='variety',y='points',data=d1)
sns.violinplot(

    x='variety',

    y='points',

    data=data[data.variety.isin(data.variety.value_counts()[:5].index)]

)
import matplotlib.pyplot as plt

fig,axarr=plt.subplots(2,1,figsize=(12,8))



data['points'].value_counts().sort_index().plot.bar(ax=axarr[0])

data['province'].value_counts().head(20).plot.bar(ax=axarr[1])
fig,axarr=plt.subplots(2,2,figsize=(12,8))

data['points'].value_counts().sort_index().plot.bar(ax=axarr[0][0],fontsize=15,color='red')

axarr[0][0].set_title("Wine Scores",fontsize=20)

data['country'].value_counts().head(20).plot.bar(ax=axarr[1][0],fontsize=18,color='red')

axarr[1][0].set_title("Origin Country",fontsize=20)

data['price'].value_counts().sort_index().plot.hist(ax=axarr[0][1],fontsize=15,color='red')

axarr[0][1].set_title("Wine Prices ",fontsize=20)

data['province'].value_counts().head(20).sort_index().plot.bar(ax=axarr[1][1],fontsize=15,color='red')

axarr[1][1].set_title("Wine Province ",fontsize=20)
