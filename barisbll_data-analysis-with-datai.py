
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
data = pd.read_csv("../input/Pokemon.csv")
data.head()
data.tail()
data.columns
data.shape
data.info()
print(data["Type 1"].value_counts(dropna=False))

#print(data["Type 1"].value_counts(dropna=False))
data.describe()
data.boxplot(column="Attack",by="Legendary")

#data.boxplot(column="Attack", by="Legendary")
data_new=data.head()
data_new
melted=pd.melt(frame=data_new,id_vars="Name",value_vars=["Attack","Defense"])
melted

melted.pivot(index="Name",columns="variable",values="value")
#Concanating Data



data1=data.head()
data2=data.tail()
conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row

#concat_data=pd.concat([data1,data2],axis=0,ignore_index=True)
#concat_data
data.dtypes
data["Type 1"]=data["Type 1"].astype("category")
data["Speed"]=data["Speed"].astype("float")
data.dtypes
data.info()
data["Type 2"].value_counts(dropna=False)

#data["Type 2"]=value_count(dropna=False)
data1=data
data1["Type 2"].dropna(inplace=True)
data1["Type 2"].describe()
assert data["Type 2"].notnull().all()
data["Type 2"].fillna("empty",inplace=True)

#data["Type 2"].fillna("empty",inplace=True)
assert data["Type 2"].notnull().all()
# 5.Part

data=pd.read_csv("../input/Pokemon.csv")
data=data.set_index("#")
data.head()
data["HP"][1]
data.HP[1]
data.loc[1,["HP"]]
#Selecting pnly some columns

data[["HP","Attack"]]
#Slicing DATA Frame

print(type(data["HP"]))
print(type(data[["HP"]]))
#Slicing and indexing series

data.loc[1:10,"HP":"Defense"]
#Reverse Slicing

data.loc[10:1:-1,"HP":"Defense"]
data.loc[1:10,"Speed":] # From something to end
#Filtering Data Frames
#Building boolean series

boolean = data["HP"]>200
data[boolean]
#Combining Filters

first_filter=data.HP>200
second_filter=data.Speed>35
data[first_filter & second_filter]
data.HP[data.Speed>150]
data4=data
data4.set_index(data.Speed)
#Transforming Data
#Plain Python Function

def div(n):
    return n/2
data.HP.apply(div)
data.HP.apply(lambda n:n/2)
data.HP
data["total_power"]=data.Attack + data.Defense
data.total_power
#Index objects and labeled Data
print(data.index.name)
data.index.name="index_name"
data
# Overwrite index
# if we want to modify index we need to change all of them.
data.head()
# first copy of our data to data3 then change index 
data3=data.copy()
data3.head()
# lets read data frame one more time to start from beginning
data = pd.read_csv('../input/Pokemon.csv')
data.head()
# As you can see there is index. However we want to set one or more column to be index
#Setting index type 1 is outer type 2 is inner index

data1=data.set_index(["Type 1","Type 2"])
data1.head(100)
#Pivoting Data Frames
dic={"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df=pd.DataFrame(dic)
df
#pivoting
df.pivot(index="treatment",columns="gender",values="response")
#Steacking and Unstacking Data

df1=df.set_index(["treatment","gender"])
df1


# level determines indexes
df1.unstack(level=0)
df1.unstack(level=1)
df
#Melting Data Frames
# df.pivot(index="treatment",columns = "gender",values="response")
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
#Categoricals and Groupby
df.groupby("treatment").mean()
df.groupby("treatment").age.max()
# Or we can choose multiple features
df.groupby("treatment")[["age","response"]].min()
df.info()
# as you can see gender is object
# However if we use groupby, we can convert it categorical data. 
# Because categorical data uses less memory, speed up operations like groupby
df["gender"] = df["gender"].astype("category")
df["treatment"] = df["treatment"].astype("category")
df.info()





