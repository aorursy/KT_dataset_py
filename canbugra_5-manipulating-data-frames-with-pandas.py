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
data = pd.read_csv('../input/Pokemon.csv')

data = data.set_index("#")

data.head(5)

data.HP[1]

data["HP"][1]

data.loc[1,["HP"]]
#selecting only some columns

data[["HP","Attack"]].head(5)
print(type(data["HP"]))
print(type(data[["HP"]]))
#Slicing and Indexing Series

data.loc[:10,"HP":"Defense"]
#reverse slicing

data.loc[10:1:-1,"HP":"Defense"]
#Taking first 20 element to find max_Hp and corresponding pokemon



HP = data["HP"][:20] #Takes first 20 element

#HP = data.loc[:20,"HP"] #Takes elements from index 1 to index 20

list1 = HP.tolist()



index_max = [];

index_data = [];

HP_max = 0;

for x in range(len(list1)):

    if (list1[x] > HP_max):

        HP_max = list1[x]

for x in range(len(list1)):

    if (list1[x] == HP_max):

        index_max.append(x)

        index_data.append(x+1)

print("Max HP is ",HP_max)

print("Index of pokemon ",index_max)

data.loc[index_max]

print(data.index.name)

data.index.name = "index name"

data.head()
data3 = data.copy()

data3.index = range(100,900,1)

data3.head()
data = pd.read_csv('../input/Pokemon.csv')

data.head()
#Setting Index

data1 = data.set_index(["Type 1","Type 2"])

data1.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],

       "age":[15,4,72,65]}

df = pd.DataFrame(dic)

df



#pivoting

df.pivot(index = "treatment",columns = "gender",values = "response")
df1 = df.set_index(["treatment","gender"])

df1
#level determine indexes

df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)

df2
#df.pivot(index = "treatment",columns = "gender",values = "response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df.groupby("treatment").mean() 
df.groupby("treatment").age.max()
df.groupby("treatment")[["age","response"]].min()