# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon.csv')
data.columns
data = pd.read_csv('../input/pokemon.csv')
data.set_index("#", inplace = True)
data.head()
data = pd.read_csv('../input/pokemon.csv')
data.set_index(["#","Name"], inplace=True, append = True, drop =False)
data.head()
data = pd.read_csv('../input/pokemon.csv')
data.set_index(["#","Name"], inplace=True, append = False, drop =True)
data.head()
data = pd.read_csv('../input/pokemon.csv')
data["Defense"][5]
# you may use this code incase
data.Defense[5]
data.loc[5,["Defense"]]
data[["Attack","Defense","HP"]]
data.tail(10)
# Difference between selecting columns: series and dataframes
print(type(data["Speed"])) #Series
print(type(data[["Speed"]])) #DataFrame
# select all columns for rows of index values 0 and 10
data.loc[[0, 10], :]
data.loc[5,['Type 1','Type 2','HP']]
#only Type 1,Type 2, HP value for 5. row
data.loc[1:5,'Type 1':'Speed']
data.loc[[1,5,7,10],'Speed']
data.loc[1:20,:'Defense']
value = data.Attack > 160
#print(value)
data[value]
#Now use two filter
filter_one = data.Legendary == True
filter_two = data.Attack > 160
data[filter_one & filter_two]


data.Name[data.Attack<25]
def f(n):
    return n+50
data.Defense.apply(f)
data.Defense.apply(lambda n:n/5)
data['New_Column'] = data.Attack+data.Defense
data.head()
print(data.index.name)
data.index.name="index"
data.head(8)
data_new = data.copy()
data_new.index = range(0,1600,2)
data_new.head()
dic = {"color":["red","green","blue","pink"],"mean":["kırmızı","yeşil","mavi","pembe"],"year":["1994","1950","1985","2000"]}
df = pd.DataFrame(dic)
df
df.pivot(index = "year",columns = "color",values = "mean")
a = df.set_index(["color","mean"])
a
a.unstack(level=0)
a.unstack(level=1)
b = a.swaplevel(0,1)
b
pd.melt(a,id_vars="year",value_vars=["color","mean"])

data.groupby("HP").mean().head()

data.groupby("HP").Speed.max().head()
