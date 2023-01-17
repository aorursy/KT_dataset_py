# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data.info()
data.corr()
f, ax = plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)

plt.show()
data.head()
#Line Plot

data.Speed.plot(kind = 'line', color = 'g', label = 'Speed', linewidth = 1, alpha = 0.5, grid = True, linestyle = ':')

data.Defense.plot(color = 'r', label = 'Defense', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
#Scatter Plot

#x=Attack ,y=Defense

data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5, color = 'red')

plt.xlabel('Attack')

plt.ylabel('Defense')

plt.title('Attack Defense Scatter Plot') #title of plot

plt.show()
#Histogram

#bins = number of bar in figure

data.Speed.plot(kind = 'hist', bins = 50, figsize = (8, 8))

plt.show()
# clf() = cleans it up again you can start a fresh 

data.Speed.plot(kind = 'hist', bins = 50)

plt.clf()
#There is two type libraries in Pandas..



#-Seriler = data[]

#-Dataframe = data[[]]

series = data['Defense'] 

print(type(series))

data_frame = data[['Defense']] 

print(type(data_frame))
#Filtering Pandas Data Frame

x = data['Defense']>200 #There are only 3 pokemons who have higher defense value than 200

data[x]
#Filtering Pandas with logical_and

#There are only 2 pokemons who have higher defense value than 200 and higher attack value than 100

data[np.logical_and(data['Defense']>200, data['Attack']>100)]
#For loop

for index, value in data[['Attack']][0:2].iterrows():

    print(index, " : ", value)
# List comprehension example with Pandas.

threshold = sum(data.Speed)/len(data.Speed)

print('threshold',threshold)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10, ["speed_level", "Speed"]]
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')

data.head()
data.columns
data.shape

data.info()
print(data['Type 1'].value_counts(dropna=False))
data.describe() # ignore null entries
data.boxplot(column='Attack', by = 'Legendary')

plt.show()
data_new = data.head()

data_new
#melting

melted = pd.melt(frame=data_new, id_vars = 'Name', value_vars = ['Attack', 'Defense'])

melted
melted.pivot(index='Name', columns = 'variable', values='value')
data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1, data2], axis=0, ignore_index=True)

conc_data_row
data3 = data['Attack'].head()

data4 = data['Defense'].head()

conc_data_col = pd.concat([data3, data4], axis=1)

conc_data_col
data.dtypes
#Lets convert object to categorical and int to float

data['Type 1'] = data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')
data.dtypes
data.info()
#Lets check Type 2

data["Type 2"].value_counts(dropna=False)

#There are 386 NAN values
#Lets drop NAN values

data1 = data

data1["Type 2"].dropna(inplace = True)
assert 1==1
assert data['Type 2'].notnull().all() 

#return error because it is false
assert data["Type 2"].notnull().all()

#returns nothing bacause we drop all NAN values
data["Type 2"].fillna('empty', inplace = True)

assert data.columns[1] == 'Name'
# Plotting all data 

data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot()
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="Attack",y = "Defense")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)
fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) 

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
import warnings

warnings.filterwarnings("ignore")

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

data2= data2.set_index("date")

data2 
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
#Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months

data2.resample("M").mean().interpolate("linear")

# read data

data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')

data= data.set_index("#")

data.head()
#indexing using square bracket

data["HP"][1]
#using column attribute and row label

data.HP[1]
#using loc accessor

data.loc[1,["HP"]]
#Selecting only some columns

data[["HP","Attack"]]
print(type(data["HP"]))     # series

print(type(data[["HP"]]))   # data frames
#slicing and indexing series

data.loc[1:10,"HP":"Defense"]
# Reverse slicing 

data.loc[10:1:-1,"HP":"Defense"] 
#from something to end

data.loc[1:10,"Speed":]
#creating boolean series

boolean = data.HP > 200

data[boolean]
#combining filters

first_filter = data.HP > 150

second_filter = data.Speed > 35

data[first_filter & second_filter]
#filtering column based others

data.HP[data.Speed<15]
#plain python functions

def div(n):

    return n/2

data.HP.apply(div)
#we can also use lambda function

data.HP.apply(lambda n : n/2)
#defining column using other columns

data["total_power"] = data.Attack + data.Defense

data.head()
print(data.index.name)

data.index.name = "index_name"

data.head()
# Overwrite index

data.head()

data3 = data.copy()

data3.index = range(100,900,1)

data3.head()
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')

data.head()
# Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["Type 1","Type 2"]) 

data1.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")

df1 = df.set_index(["treatment","gender"])

df1
# level determines indexes

df1.unstack(level=0)
df1.unstack(level=1)

#change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
pd.melt(df,id_vars="treatment",value_vars=["age","response"])

#we will use df

df
df.groupby("treatment").mean()   # mean is aggregation / reduction method
#we can only choose one of the feature

df.groupby("treatment").age.max() 
#Or we can choose multiple features

df.groupby("treatment")[["age","response"]].min() 
df.info()