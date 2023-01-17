# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2016.csv')

data.head()
data = data.set_index('Happiness Rank')

data.head()
data['Happiness Score'][1]
data.Region[10]
data.loc[1,['Region']]
data[['Region', 'Happiness Score']]
print(type(data["Region"]))     # series

print(type(data[["Region"]]))   # data frames
data.loc[1:10,"Region":"Freedom"] 
data.loc[10:1:-1,"Region":"Freedom"] 
data.loc[1:10,"Trust (Government Corruption)":] #From a column to end
boolean = data['Happiness Score'] > 7.000

data[boolean]
boolean1 = data['Region'] == 'Western Europe'

boolean2 = data['Freedom'] < 0.5

data[boolean1 & boolean2]
data.Region[data.Generosity < 0.1]
# Plain python functions

def div(n):

    return n/2

data['Happiness Score'].apply(div)
#lambda function

data['Generosity'].apply(lambda n : n/2)
# Defining column using other columns

data["Confidence"] = (data['Lower Confidence Interval'] + data['Upper Confidence Interval']) / 2

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

data1 = data.copy()

# lets make index start from 100. It is not remarkable change but it is just example

data1.index = range(100,257,1)

data1.head()
# lets read data frame one more time to start from beginning

data = pd.read_csv('../input/2016.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["Happiness Score","Family"]) 

data1.head(100)

# data1.loc["Fire","Flying"] # howw to use indexes
dic = {"cure":["A","A","B","B"],"gender":["F","M","F","M"],"response to cure":[10,45,5,9],"age":[23,18,53,49]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="cure",columns = "gender",values="response to cure")
df1 = df.set_index(["cure","gender"])

df1

# lets unstack it
df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
# df.pivot(index="cure",columns = "gender",values="response to cure")

pd.melt(df,id_vars="cure",value_vars=["age","response to cure"])
df
# according to cure take means of other features

df.groupby("cure").mean()   # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
df.groupby("cure").age.max() 
df.groupby("cure")[["age","response to cure"]].min() 
df.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

df["gender"] = df["gender"].astype("category")

df["cure"] = df["cure"].astype("category")

df.info()