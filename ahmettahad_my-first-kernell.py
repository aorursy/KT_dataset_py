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

import matplotlib.pyplot as plt



df = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
df.info

df.head()

df.columns
df.corr()
f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(df.corr(),annot = True,linewidths=1.5,linecolor="blue",vmin=-1,vmax=1,cmap="YlGnBu")

plt.show()
sns.pairplot(df.loc[0:,['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']])

plt.show()
df.NA_Sales.plot(kind='line',color='g',alpha=0.4,linewidth=1,grid=True)

df.EU_Sales.plot(kind='line',color='b',alpha=0.4,linewidth=1,grid=True)

df.JP_Sales.plot(kind='line',color='r',alpha=0.4,linewidth=1,grid=True)

plt.legend(loc="upper right")

plt.xlabel("Sales")

plt.ylabel("--")

plt.title("Sales")

plt.show()
fig,ax = plt.subplots(figsize=(8,5))

df['Genre'].value_counts(sort=False).plot(kind='bar',ax=ax,rot =65,figsize=(9,9))

plt.title('Genre Distribution',fontsize=15)

plt.xlabel('Genre',fontsize=15)

plt.ylabel('Sales',fontsize=15)
plt.plot(df.Rank,df.Global_Sales,color="green",alpha=.7)

plt.title('Ranking Effect on Sales',fontsize=15)

plt.ylabel("Global Sales")

plt.xlabel("Rank")

plt.show()
x = df.Year

y = df.Global_Sales

plt.bar(x,y)

plt.title('Years Effect of Rank',fontsize=15)

plt.xlabel("x")

plt.ylabel("y")

plt.show()