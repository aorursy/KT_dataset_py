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
data = pd.read_csv('../input/pokemon/Pokemon.csv')

data.head()
data = data.set_index('#')

data.head()
# indexing using square brackets

data["HP"][1]
# using column attribute and row label

data.HP[1]
# using loc accessor

data.loc[1:4,["HP"]]

# Selecting only some columns

data[["HP","Attack"]].head(8)
# Difference between selecting columns: series and dataframes

print(type(data["HP"]))     # series

print(type(data[["HP"]]))   # data frames
# Slicing and indexing series

data.loc[1:10,"Attack":"Defense"]
# Reverse slicing 

data.loc[10:1:-1,"HP":"Defense"]
# From something to end

data.loc[1:10,"Speed":] 
# Creating boolean series

boolean = data.HP > 200

data[boolean]
# Combining filters

first_filter = data.HP > 150

second_filter = data.Speed > 35

data[first_filter & second_filter]
# Filtering column based others

data.HP[data.Speed<15]
# Plain python functions

def div(n):

    return n/2

data.HP.apply(div) # in this case with apply() function, we run our div(n) function for every sample of 'data.HP' 

#                    I mean, every sample to be sent to our function as  an argument
# Or we can use lambda function

data.HP.apply(lambda n : n/2)
# Defining column using other columns

data["total_power"] = data.Attack + data.Defense

data.head()
# our index name is this:

print(data.index.name)

# lets change it

data.index.name = "index_name"

data.head()
# Overwrite index

# if we want to modify index we need to change all of them.

data.head()

# first copy of our data to data3 then change index 

data3 = data.copy()

# lets make index start from 100. It is not remarkable change but it is just example

data3.index = range(0,800,1)

data3.head()
# lets read data frame one more time to start from beginning

data = pd.read_csv('../input/pokemon/Pokemon.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["Type 1","Type 2"]) 

data1.head(100)

# data1.loc["Fire","Flying"] # howw to use indexes
dic = {"id":[0,1,2,3],"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
df = df.set_index("id")

df.head()
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1
# level determines indexes

df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
# according to treatment take means of other features

df.groupby("treatment").mean()   # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("treatment").age.max() 
# Or we can choose multiple features

df.groupby("treatment")[["age","response"]].min() 