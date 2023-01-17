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
data = pd.read_csv("../input/pokemon.csv")
data.columns
data.head()
# data.tail()
data.index = range(1,802,1) # index starts from 1
data.name[0:3] # however, zero is the index that's exclusive. show data 1st to 3rd index.

data["type1"][5] # 5th index of 'type1'
data.loc[1,"type1":"type2"]  # 1st raw of type1 to type2
data[["name","speed"]] # show name and speed throughout the list
filtered = data.speed > 150
data[filtered]
filter1 = data.attack > 150
filter2 = data["defense"] > 120
data[filter1 & filter2]
data.speed[data.attack>180] # speed of pokemons of the ones having greater attack than 180.
def prod(n):
    return n**2
data["attack"].apply(prod) # squares of attack column 
data["full speed"] = data.sp_attack + data.sp_defense 
data.head()
data = pd.read_csv("../input/pokemon.csv")
print(data.index.name)  # we dont have an index name yet.
data.index.name = ["benimsin"] # name of my index is " benimsin"
data1 = data.set_index("type1")  # setting another column as index
# data.index.name = data["smth"]
data1.head()
data1 = data.set_index(["type1","type2"])
data1.head()
dic= {"sex":["F","M","M","F"],"sizes":[13,17,12,19],"smt":["x","x","y","y"]}
df = pd.DataFrame(dic)
df.index = range(1,8,2)  # index counts two each
df
df.pivot(index = "sex",columns= "smt",values="sizes")
df2 = df.set_index(["smt","sex"])
df2
df2.unstack(level=0)    # remove first index column
df3 = df2.swaplevel(0,1)
df3
pd.melt(df,id_vars="sex",value_vars=["smt","sizes"])

df.groupby("sex").mean()
df.groupby("smt").sizes.max()