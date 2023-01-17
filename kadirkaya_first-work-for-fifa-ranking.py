

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #visualization tool
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/fifa_ranking.csv') # import data
data.info()
data.describe()
data.columns
data.head(10)
data.corr() # correlation of data
#correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True ,linewidth=5, fmt= '.1f',ax=ax)
plt.show()
# LİNE PLOT
# x= previous_points,y=rank_change
data.previous_points.plot(kind='line',color='red',label="previous_points",linewidth=2,alpha=0.5,grid=True,linestyle=':')
data.rank_change.plot(kind='line',color='blue',label="rank_change",linewidth=2,alpha=0.5,grid=True,linestyle='--')
plt.legend(loc='upper left') # loc=location
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title("Line Plot")
plt.show()
#Scatter Plot
#x=rank,y=previous_points
data.plot(kind='Scatter',x='rank',y='previous_points',alpha=1,color='blue',linewidth=0.2)
plt.xlabel('Rank')
plt.ylabel('Previous_points')
plt.title('Rank PP Scatter Plot')
plt.show()

#Histogram
data.previous_points.plot(kind='hist',bins=20,figsize=(10,10))
plt.show()
#Fİltering
series = data['previous_points']        # data['Defense'] = series
print(type(series))
data_frame = data[['previous_points']]  # data[['Defense']] = data frame
print(type(data_frame))
#Filtering with dataframe
x=data['previous_points']>1900
data[x]
# 2 - Filtering numpy with logical_and
data[np.logical_and(data['previous_points']>1700, data['rank']<100 )]
#we can also use '&' for filtering.
data[(data['previous_points']>1850) & (data['rank']<100)]
import builtins
dir(builtins)
def cube():
    
    def add():
        x=10
        y=11
        z=x+y
        return z
    return add ()**3
print(cube())
#Default Arguments
def f(a,b=10,c = 4): #Please attention,hear is that non-default follows default.
    z=a+b*c
    return z
print(f(10))
# İf you wanna use different numbers that this function.
print(f(2,4,5))
#Threshold Value
threshold=sum(data.previous_points)/len(data.previous_points)
data["value_level"]=["high" if i>threshold else "low" for i in data.previous_points]
data.loc[:20,["value_level","previous_points"]]

data=pd.read_csv("../input/fifa_ranking.csv")
data.tail()
data.columns
data.shape #Data's row and columns
data.info()
print(data['confederation'].value_counts(dropna =False))#dropna=null is deleted
data.describe()
new_data=data.head()
new_data
melted= pd.melt(frame=new_data,id_vars='country_full',value_vars=['previous_points','country_abrv'])
melted

#Reverse Melted
melted.pivot(index = 'country_full', columns = 'variable',values='value')
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
data1 = data['country_full'].head()
data2= data['previous_points'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col
data["country_full"].value_counts(dropna =False)
#Zip Method

country = ["Turkey","Hungary"]
population = ["100","120"]
list_label = ["country","population"]
list_col = [country,population]
zipp = list(zip(list_label,list_col))
data_dict = dict(zipp)
df = pd.DataFrame(data_dict)
df
#add column

df["Capital"]=["Ankara","Budapest"]
df
#add income
df["income"]=100
df
#PLOT
data.columns

data1=data.loc[:,["rank","previous_points","rank_change"]]
data1.plot()


#Subplots
data1.plot(subplots=True)
plt.show()
#Scatter Plot
data1.plot(kind="scatter",x="rank",y="rank_change")
plt.show()
#Histogram Plot
data1.plot(kind = "hist",y = "previous_points",bins = 100,range= (0,300),normed = True)
#Normed=Normalization on data

fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "previous_points",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "previous_points",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
data.describe()
time_list=["1994-03-09","1997-09-09"]
print(type(time_list))
#list(class) convert to datetime(class)
datetime_object=pd.to_datetime(time_list)
print(type(datetime_object))


data2=data.head()
date_list=["1994-03-09","1994-06-20","2015-01-04","1997-09-09","1997-08-23"]
datetime_object=pd.to_datetime(date_list)
data2["date"]=datetime_object
data2=data2.set_index("date")
data2
#Selecting with type of datetime
print(data2.loc["1994-03-09"])
print(data2.loc["1994-06-20":"1997-08-23"])
#Resampling Pandas Time Series
# M=Month A=Year
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").mean().interpolate("linear") #Linear İnterpolate
data2.resample("M").first().interpolate("linear")
# read data
data=pd.read_csv("../input/fifa_ranking.csv")
data.head()
data["country_full"][0]
data.country_full[0]
data.loc[0:5,["country_full"]]
data[["rank","country_full"]]
#SLICING DATAFRAME
print(type(data["country_full"])) #Series
print(type(data[["country_full"]]))#DataFrame


data.loc[0:10,"rank":"country_full"]
data.loc[10:1:-1,"rank":"country_full"]#sorting from reverse
data.loc[1:10,"country_full":]
boolean=data.previous_points >1900
data[boolean]
first_filter=data.previous_points>1800
second_filter=data.rank_change>-1
data[first_filter&second_filter]
data.rank_change[data.previous_points>1750]
def div(n):
    return n/2

data.previous_points.apply(div)

data.previous_points.apply(lambda n:n/2)
data["totals"]=data.previous_points+data.rank_change
data.head()
# our index name is this:
print(data.index.name)
# lets change it
data.index.name = "index_name"
data.head()
# We can change number range..
data3=data.copy()
data3.index=range(100,57893,1)
data3.head()
data=pd.read_csv("../input/fifa_ranking.csv")
data.head()
#Setting index
data1 = data.set_index(["rank","previous_points"]) 
data1.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[19,55,52,90],"age":[15,40,78,65]}
df = pd.DataFrame(dic)
df
df.pivot(index="treatment",columns="gender",values="response")

df1=df.set_index(["treatment","gender"])
df1
df1.unstack(level=0)

df1.unstack(level=1)

# To Change place of indexes
df2=df1.swaplevel(0,1)
df2
#Reverse of pivoting
df
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
df.groupby("treatment").mean()
df.groupby("treatment").age.max()
df.groupby("treatment")[["age","response"]].min() 

df.info()