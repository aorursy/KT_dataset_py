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
data=pd.read_csv('../input/Pokemon.csv')
data=data.set_index("#")
data.head()
data["Defense"][4]
data.Speed[1]
data.loc[3,["Attack"]]
data[["HP","Sp. Atk"]]
print(type(data["Attack"]))
print(type(data[["Attack"]]))
data.loc[1:20,"Total":"Defense"]
data.loc[20:1:-1,"Total":"Defense"]
data.loc[1:20,"HP"]
boolean=data.Defense>150
data[boolean]
first_filter=data.Defense>150
second_filter=data.Attack>100
data[first_filter & second_filter]
data.Attack[data.Defense>170]
def div(n):
    return n*2
data.Attack.apply(div)
data["max_power"]=data.HP+data.Speed
data.head()
print(data.index.name)
data.index.name="id"
data.head()
data.head()
data3 = data.copy()
data3.index = range(100,900,1)
data3.head()

data=pd.read_csv('../input/Pokemon.csv')
data.head()
data1=data.set_index(["Attack","Defense"])
data1.head()
#Pivoting Data
dic={"blood_type":["0","A","B","AB"],"Rh":["+","-","-","-"],"Age":[15,25,35,45],"Gender":["F","M","M","F"]}
df=pd.DataFrame(dic)
df
df.pivot(index="Age",columns="blood_type",values="Rh")
df1=df.set_index(["Age","Gender"])
df1
df1.unstack(level=0)
df1.unstack(level=1)
df2=df1.swaplevel(0,1)
df2
pd.melt(df,id_vars="Age",value_vars=["blood_type","Rh"])
df
df.groupby("Gender").mean()
df.groupby("blood_type").Age.max() 
df.groupby("Gender")[["Age","blood_type"]].min()
df["Gender"]=df["Gender"].astype("category")
df.info()