# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(10)
data.columns

# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='lower right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter',x='Attack',y='Defense',alpha = 0.5,color = 'red')
plt.xlabel('Attack')              # label = name of label
plt.ylabel('Defence')
plt.title('Attack Defense Scatter Plot')            # title = title of plot
# Histogram 
data.Speed.plot(kind='hist',bins=50,figsize=(12,12))
plt.show()
data.Speed.plot(kind='hist',bins=50)
plt.clf()
dictionary = {'germany' : 'munih','Türkiye' : 'Ankara'}
print(dictionary.keys())
print(dictionary.values())
series = data['Defense']
print(type(series))
data_frame = data[['Defense']]
print(type(data_frame))
# Filtering 1
x = data['Defense']>175
data[x]
# Filtering 2
data[np.logical_and(data['Defense']>150, data['Attack']>100 )]
# Filter 3
data[(data['Defense']>175) & (data['Attack']>100)]
for key,value in dictionary.items():
    print(key,' : ',value)
for index,value in data[['Attack']][0:10].iterrows():
    print(index,' : ',value)
import builtins
dir(builtins)
# Lambda function
cube = lambda x: x**3
print(cube(3))
number_list = [4,5,6]
y = map(lambda x:x**3,number_list)
print(list(y))
# Zip
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
# un zip
un_zip = zip(*z_list)
print(un_zip)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
# list comprehension
num1 = [2,3,4]
num2 = [i *2 for i in num1]
print(num2)
num3 = [10,15,5]
num3_bool = [True if i > 10 else False if i < 10 else None for i in num3]
print(num3_bool)
threshold = sum(data.Speed)/len(data.Speed)
data['speed_level'] = ['high' if i > threshold else 'low' for i in data.Speed]
data.loc[:10,['speed_level','Speed']]
data.head()
data.tail()
data.columns
data.shape
data.info()
print(data['Type 1'].value_counts(dropna=False))
data.describe()
data.boxplot(column='Attack',by='Legendary')
data_new = data.head()
data_new
# Melt
melted = pd.melt(frame=data_new,id_vars='Name',value_vars=['Attack','Defense'])
melted
# Pivot
melted.pivot(index='Name',columns='variable',values='value')
# Concet
data1 = data.head()
data2 = data.tail()
conc_data = pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data
data1 = data['Attack'].head()
data2 = data['Defense'].head()
conc_data = pd.concat([data1,data2],axis=1)
conc_data
data.dtypes
data['Type 2'].value_counts(dropna=False)
data['Type 2'].dropna(inplace=True)
data
assert data['Type 2'].notnull().all()
data1 = data['Type 2'].fillna('empty',inplace=True)
print(data1)
# data frames from dictionary
country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zippend = list(zip(list_label,list_col))
data_dict = dict(zippend)
df = pd.DataFrame(data_dict)
df
df["capital"] = ["madrid","paris"]
df
df['income']=0
df
data1 = data.loc[:,["Attack","Defense","Speed"]]
data1.plot()
data1.plot(subplots=True)
data.plot(kind='scatter',x='Attack',y='Defense')
plt.show()
data1.plot(kind='hist',y='Defense',bins=50,range=(0,250),normed=True)
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt.show()
data.describe()
time_list = ["1992-03-08","1992-04-12"]
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
# close warning
import warnings
warnings.filterwarnings("ignore")
# In order to practice lets take head of pokemon data and add it a time list
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
# lets make date as index
data2= data2.set_index("date")
data2 
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample('M').first().interpolate("linear")
data2.resample("M").first().interpolate("linear")
data = pd.read_csv('../input/pokemon.csv')
data= data.set_index("#")
data.head()
data["HP"][1]
data.HP[1]
data.loc[1,["HP"]]
data[["HP","Attack"]]
print(data.HP.apply(lambda z:z/2))
print(data.index.name)
data.index.name = 'index'
print(data.index.name)
data.head()
# first copy of our data to data3 then change index 
data3 = data.copy()
# lets make index start from 100. It is not remarkable change but it is just example
data3.index = range(100,900,1)
data3.head()
data1 = data.set_index(["Type 1","Type 2"]) 
data1.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
df.pivot(index="treatment",columns="gender",values="response")
df1 = df.set_index(["treatment","gender"])
df1
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)
df2
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df.groupby("treatment").mean()
df.groupby("treatment")[["age","response"]].min()
df["gender"] = df["gender"].astype("category")
df.info()
df
df["treatment"] = df["treatment"].astype("category")
df.info()
df