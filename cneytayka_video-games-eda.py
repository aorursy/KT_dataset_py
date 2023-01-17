# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline
import warnings
warnings.filterwarnings('ignore') 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
games=pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
games.head()
games.info()
games.describe()
# NA_Sales vs EU_Sales
g10=games.iloc[:10,:]
g10[["NA_Sales","EU_Sales"]].groupby(["NA_Sales"],as_index=False).mean().sort_values(by="EU_Sales",ascending=True)
g10=games.iloc[:10,:]
g10[["EU_Sales","JP_Sales"]].groupby(["EU_Sales"],as_index=False).mean().sort_values(by="JP_Sales",ascending=True)
g10=games.iloc[:10,:]
g10[["Other_Sales","Global_Sales"]].groupby(["Other_Sales"],as_index=False).mean().sort_values(by="Global_Sales",ascending=True)
games.head()
games.columns[games.isnull().any()]
games.isnull().sum()
games[games["Publisher"].isnull()]
games.head()
#games.Genre.value_counts()
#genre nin ve Global Salenın type baktık
#games.info()

games.Genre.unique()
#Bar plot 
#example:Genre vs Global_sales

area_list=list(games["Genre"].unique())
area_global_ratio=[]

for i in area_list:
    
    x=games[games["Genre"]==i]
    
    area_global_rate=sum(x.Global_Sales)/len(x)
    
    area_global_ratio.append(area_global_rate)
    
#Sorting

data=pd.DataFrame({"area_list":area_list,"area_global_ratio":area_global_ratio})
new_index=(data["area_global_ratio"].sort_values(ascending=False)).index.values
sorted_data=data.reindex(new_index)

#visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data["area_list"],y=sorted_data["area_global_ratio"])

plt.xticks(rotation=45)

plt.xlabel("Genre")
plt.ylabel("Global Sales")
plt.title("Global Sales of Genre")
plt.show()

games.head()
g = sns.jointplot(games.NA_Sales,games.EU_Sales, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
y = games.groupby(['Year']).sum()
y = y['Global_Sales']
x = y.index.astype(int)

plt.figure(figsize=(12,8))
ax = sns.barplot(y = y, x = x)
ax.set_xlabel(xlabel='$ Millions', fontsize=16)
ax.set_xticklabels(labels = x, fontsize=12, rotation=90)
ax.set_ylabel(ylabel='Year', fontsize=16)
ax.set_title(label='Game Sales in $ Millions Per Year', fontsize=20)
plt.show();
#plt.style.use("classic")
name=games.Name.value_counts()

plt.figure(figsize=(15,10))
ax=sns.barplot(x=name[:10].values,y=name[:10].index)
plt.xticks(rotation=90)
plt.xlabel("Game Name",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.title("Game Name Top 10",color="red",fontsize=20)
#başta datamızın columlarına tekrar göz gezdirdik
#games.head()
sns.set_style("dark")
sns.despine()
table = games.pivot_table('Global_Sales', index='Platform', columns='Year', aggfunc='sum')
platforms = table.idxmax()
sales = table.max()
years = table.columns.astype(int)
data = pd.concat([platforms, sales], axis=1)
data.columns = ['Platform', 'Global Sales']

plt.figure(figsize=(15,10))
ax = sns.pointplot(y = 'Global Sales', x = years, hue='Platform', data=data, size=15)
ax.set_xlabel(xlabel='Year', fontsize=16)
ax.set_ylabel(ylabel='Global Sales Per Year', fontsize=16)
ax.set_title(label='Highest Total Platform Revenue in $ Millions Per Year', fontsize=20)
ax.set_xticklabels(labels = years, fontsize=12, rotation=50)
plt.show();
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(games.corr(), annot=True, linewidths=0.5,linecolor="blue", fmt= '.1f',ax=ax)
plt.show()
sns.pairplot(data)
plt.show()