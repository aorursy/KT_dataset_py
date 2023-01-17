# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read data

data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data = data.set_index('id')

data.head()
# indexing using square brackets

data["price"][2539]
# Using column attribute and row label

data.price[2539]
# using lco accessor

data.loc[2539,["price"]]
# Selecting only some columns

data[["price","room_type"]]
# Difference between selecting columns: series and dataframes

print(type(data["price"]))

print(type(data[["price"]]))
# slicing and indexing series

data.loc[2539:25000,["price","room_type"]]
# reverse slicing

data.loc[24143:2539:-1,"price","minimum_nights"]
# From something to end

data.loc[2539:25000,"price":]
# Creating boolean series

boolean = data.price >6000

data[boolean]
# Combining filters

first_filter = data.price >6000

second_filter = data.minimum_nights >15

data[first_filter & second_filter]
# Filtering column based others

data.price[data.minimum_nights<30]
# Plain python functions

def div(n):

    return n/2

data.price.apply(div)
# or we can use lambda function

data.price.apply(lambda n: n/2)
# Defining column using other columns

data["room_fee_per_day"] = data.price / data.minimum_nights

data.head()
# our index name is this:

print(data.index.name)

#lets change it

data.index.name = "id_number"

data.head()
# Overwrite index

# if we want to modify index we need to change all of them

data.head()

# first copy of our data to data3 then change index

data3 = data.copy()

#lets make index start from 100. 

data3.index = range(1,48896,1)

data3.head()
# lets read data frame one more time to start from beginning

data1 = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data1.head(100)
# Setting index: neighbourhood is outer minimum_nights is inner index

data1 = data.set_index(["neighbourhood","neighbourhood_group"])

data1.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
#pivoting

df.pivot(index ="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df
# level determines indexes

df1.unstack(level =0)
df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
# df.pivot(index ="treatment",columns = "gender",values="response")

pd.melt(df, id_vars ="treatment", value_vars=["age","response"])
# We will use df

df
# according to treatment take means of other features

df.groupby("treatment").mean()   # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("treatment").age.max() 
# Or we can choose multiple features

df.groupby("treatment")[["age","response"]].min() 