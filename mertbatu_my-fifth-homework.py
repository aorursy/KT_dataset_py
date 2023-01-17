import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/tmdb_5000_movies.csv')

data = data.set_index("release_date") # If the dataframw has a index number,"#" sign is used for setting index.

data.head()
data["budget"][0]
data.budget[0]
data.loc["2009-12-10",["budget"][0]] # If the index is a number, we can write data.loc[1,["budget"]] and get same result.
data[["budget","runtime"]] # Selecting some columns
print(type(data["budget"])) # series

print(type(data[["budget"]])) #data frames
data = data.reset_index() # resetting the index
data.loc[1:10,"title":"vote_count"] # Slicing the data frame. Only title between vote_count features are printing.
data.loc[10:1:-1,"title":"vote_count"] # Reverse sorting/slicing
data.loc[1:10,"popularity":] # Slicing popularity to end of the data frame.
boolean = data.vote_average > 9.5

data[boolean]
first_filter = data.vote_average > 8 # Combinig the two filters

second_filter = data.runtime > 180

data[first_filter & second_filter]
data.runtime[data.vote_average > 9.5] # Column based filter
def convert(a): # Converting to the million

    return a/10**6

data.budget.apply(convert)
data.budget.apply(lambda n:n/10**3) # Converting to the thousands using lambda
data["profit"] = data.revenue - data.budget # Adding/Defining the new column using other columns
data.head()
print(data.index.name) # Controlling the name of the index
data.index.name = "index_name" # Changing the index name.

data.head()
data3 = data.copy()

#data3.index = range(100,120,1) # In here, I face off ValueError problem I don't know why. So I applied other method for printing

data3.head()

data4 = data.loc[100:120] # Printing dataframes that are between 100-120 indexes.

data4
data = pd.read_csv('../input/tmdb_5000_movies.csv')

data.head()
data.index.name = '#'
data.head()
data.genres.name 
data1 = data.set_index(["status", "original_language", "vote_average"]) # Outer index, middle index, inner index

data1.head()
data5 = data.loc[0:4802,["status"]]

data5[data5.status != "Released"]
data5 = data1.loc[["Released", "Post Production","Rumored"]]

data5.head()
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1
# level determines indexes

df1.unstack(level=0) # inner index
df1.unstack(level=1) # outer index
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
pd.melt(df, id_vars ="treatment", value_vars = ["age", "response"])
df
df.groupby("treatment").mean()
df.groupby("treatment").age.max() # Choosing one feature
df.groupby("treatment").min()
df.groupby("treatment")[["age","response"]].min() # Choosing more than one features