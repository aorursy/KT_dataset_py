import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv('../input/Pokemon.csv')
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1]))

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")



data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1992-04-10","1992-05-10",]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets test date as index

data2 = data2.set_index("date")

data2
#Now we can select according to our date index

print(data2.loc["1992-01-10"])

print(data2.loc["1992-02-10":"1993-03-10"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data3 = pd.read_csv('../input/Pokemon.csv')

data3 = data.set_index("#")

data3.head()
# İndexing using square brackets

data["HP"][1]
# Using column attribute and row label

data.HP[2]
# Using loc accessor

data.loc[1, ["HP"]]
# Selecting only some columns

data[["HP","Attack"]]
#Difference between selecting columns : series and dataframes

print(type(data["HP"]))   # Series

print(type(data[["HP"]])) # Data Frames
# From something to end 

data.loc[1:10,"HP":]
#Creating boolean series

boolean = (data.HP > 200)

data[boolean]
# Combining filters

first_filter = data.HP > 100

second_filter = data.Speed > 40

data[first_filter & second_filter]
# Filtering column based others

data.HP[data.Speed < 15]

# Plain python functions

def div(n):

    return n/2

data.HP.apply(div)

# or we can use lambda function with apply command

data.HP.apply(lambda n : (n*4)/2)
#Defining column by using other columns

data["total_power"] = data.Attack + data.Defense

data.loc[1:10,"HP":]





# our index name is this:

print(data.index.name)

# lets change it

data.index.name = "ID" # İndex name

data.head()
#overwrite index 

# if we want to modify index we need to change all of them

data.head

#first copy our data to data10 then change index

data10 = data.copy()

#lets make index start from 100. it is not remarkable changing but it is just example

data10.index = range(100,900,1)

data10.head()
# lets read data frame one more time to start from beginning

data = pd.read_csv('../input/Pokemon.csv')

data.head()

#Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["Type 1","Type 2"])

data1.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
#Pivoting

df.pivot(index="treatment",columns = "gender",values="response")

    
#according to treatment take means of other features

df.groupby("treatment").mean()
df1 = df.set_index(["treatment","gender"])

df1

# lets unstack it
# level determines indexes

df1.unstack(level=0)
#Melting Data Frames

pd.melt(df,id_vars="treatment",value_vars=["age","response"])

df.groupby("treatment").mean()
#we can only choose one of the feature

df.groupby("treatment").age.min()

df.groupby("treatment").age.max()
df.groupby("treatment")[["age","response"]].min()
df.info()