import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

df = pd.read_csv('../input/NBA_player_of_the_week.csv')
#gives top 10 rows
df.head(10)
#gives last 5 rows
df.tail()
# gives information about dataframe. Like rows name, type, count and using memory kb
df.info()

# gives column's names
df.columns
# gives number of rows and number of columns
df.shape
# columns renamed with lower() and replace() methods
df.columns = df.columns.str.lower().str.replace(' ','_')
df.columns
df.describe()
# frequency count about chosing column
# for example, Miami Heat has 57 players
df["team"].value_counts(dropna=False)
#visual exploratory data analysis
df.boxplot(column='age',by='team')
plt.show()
# melt method example
# frame= data which do melt
# id_vars = id in melt data
# value_vars = what we want to melt
df_new = df.tail(3)

melted = pd.melt(frame=df_new, id_vars='player',value_vars=['height','weight','age'])
melted
# pivoting data
# reverse melting

melted.pivot(index='player', columns='variable', values='value')
# concatenating data
# axis=0 by rows, axis=1 by columns, ignore_index=True assign new index
data1 = df.sample(n=10)
data2 = df.sample(n=5)
conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row

data1 = df.player.head()
data2 = df.team.head()
data3 = df.season_short.head()
conc_data_column = pd.concat([data1,data2,data3],axis=1)
conc_data_column

df.sort_values('season_short')
df.dtypes
df['conference'] = df.conference.astype('category')
df.dtypes
# missing data
df.conference.value_counts(dropna=False)
data1 = df.copy()
data1.conference.value_counts(dropna=False)
data1.conference.dropna(inplace=True)
data1

#assert is control true or false

assert 1==1 # if it is true, return nothing
assert 1==2 # if it isfalse, return error
assert data1.conference.notnull().all() #return nothing because we drop nan values
df['conference'] = df.conference.astype('object')
data1.conference.fillna('empty',inplace=True)

data1

#review pandas
#dataframe from list

country = ["Italy","England"]
population = ["12","25"]

lis_columns = ["country","population"]
lis_rows = [country,population]
zipped = list(zip(lis_columns,lis_rows))

dic = dict(zipped)

data = pd.DataFrame(dic)
data
data["capital"] = ["Roma","London"]
data
data["income"] = 0
data
#virsual exploratary

df1 = df.loc[:,["age","draft_year","season_short"]]
df1.plot()
plt.show()
df1.plot(subplots=True)
#scatter plot
df1.plot(kind="scatter",x="age",y="season_short")
plt.show()
#histogram
# frekans ölçüyor
# range bizim x eksenimiz
# normed frekansının normalize eder 0 ile 1 arasında
df1.plot(kind="hist",y="age",bins=50,range=(0,50),normed=True)
plt.show()
# histogram subplot with non cumulative and cumulative
# cumulative önceki değerleri toplayarak gidiyor
fig,axes = plt.subplots(nrows=2,ncols=1)
df1.plot(kind="hist", y="age", bins=50,range=(0,50),normed=True,ax=axes[0])
df1.plot(kind="hist", y="age", bins=50,range=(0,50),normed=True,ax=axes[1],cumulative=True)
plt.savefig('graph.png')
plt.show()
time_list=["1992-02-26","1991-08-08"]
print(type(time_list[1]))

obj_datetime = pd.to_datetime(time_list)
print(type(obj_datetime))
df2 = df.head()
date_list = ["2018-01-10","2018-02-10","2018-03-10","2017-7-5","2017-7-1"]
obj_date = pd.to_datetime(date_list)
df2["date"] = obj_date
df2 = df2.set_index("date")
df2
print(df2.loc["2018-02-10"].player)
print(df2.loc["2017-07-05":"2018-02-10"])
# resample pandas time series
df2.resample('A').mean() # resample by year and show means
df2.resample('M').mean() # resample by month and show means
df2.resample('M').first().interpolate("linear") # fill with linear values to numeric NANs
#indexing and slicing

index_list=[]
i=1
while i<=len(df):
    index_list.append(i)
    i=i+1

df["id"] = index_list
df = df.set_index("id")
df
df.team[10] #df["team"][10]
df.loc[5,"team"]
df[["player","team","season_short"]]
print(type(df["age"]))
print(type(df[["age"]]))
df.loc[3:9,"player":"team"]
df.loc[10:1:-1,"player":"season"]
df.loc[1:10,"position":]
#filtering
flt = df.seasons_in_league > 15
df[flt]

flt1 = df.seasons_in_league > 10
flt2 = df.team == "Houston Rockets"
df[flt1 & flt2]
# Transforming data

def square(x):
    return x**2

df.age.apply(square)
    

df.age.apply(lambda x:x/2)
#defining column using other columns
df["abstract"] = df.player + " - " +df.team + " - " +df.position
df.head()
print(df.index.name)
df.index.name = "index_name"

data1 = df.copy()
data1.index=range(100,1245,1)
data1.head()
# hierarchical index
data1 = df.set_index(["team","position"])
data1
#pivoting data

dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
dataframe = pd.DataFrame(dic)
dataframe
dataframe.pivot(index="treatment",columns="gender",values="response")
dataframe
#stacking and unsteaking dataframe
dataframe1 = dataframe.set_index(["treatment","gender"])
dataframe1
dataframe1.unstack(level=0)
dataframe1.unstack(level=1)
# translocation to index level
dataframe2 = dataframe1.swaplevel(0,1)
dataframe2
#melting dataframe
dataframe
pd.melt(dataframe,id_vars = "treatment",value_vars=["age","response"])
dataframe.groupby("gender").mean()
dataframe.groupby("treatment").age.max()
dataframe.groupby("gender")[["age","response"]].min()
