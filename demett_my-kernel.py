# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
data.head() #show first 5 row of data
data.columns #show data column names
data.shape #show row and column counts
data.info() #show some information about data like column names, data tyoes, non-null value counts...
#frequency of country data

print(data["country"].value_counts(dropna=False))

#if there are non values, also show them 
data.describe() #ignore null values
data.boxplot(column='release_year',by='type')

plt.show

#black line at top is max

#blue line at top is 75%

#green line is median 50%

#blue line at bottom is 25%

#black line at bottom is min

#circles are outliers..
new_data=data.head()
new_data
melted=pd.melt(frame=new_data,id_vars='title',value_vars=['type','duration'])
melted
melted.pivot(index='title',columns='variable',values='value')
data1=data.head()

data2=data.tail() #we'll concatenate these dataframes.
conc_data=pd.concat([data1,data2],axis=0,ignore_index=True)

#concatenate data1 and data2 vertically
conc_data
d1=data['title'].head()

d2=data['release_year'].head()
c_data=pd.concat([d1,d2],axis=1)

#concatenate data1 and data2 horizontally
c_data
data.dtypes
data['type'].unique()
data['type']=data['type'].astype('category')
data.dtypes
data.info()
data.isnull().sum() #How many nulls are in which column
data["director"].value_counts(dropna=False)

#There are 1969 NaN (missing) values
data.dropna(subset=['director'],axis=0,inplace=True)

#inplace=True means dont have to assign new variable
#check with assert statement

assert data["director"].notnull().all() # if it's true, return nothing

#if it's false, return error



# assert 1=1 return nothing

# assert 1=2 return error
data
data["cast"].fillna('empty',inplace=True)
#check with assert statement

assert data["cast"].notnull().all() # if it's true, return nothing

#if it's false, return error



# assert 1=1 return nothing

# assert 1=2 return error
data["cast"].value_counts(dropna=False)

#There isn't any NaN (missing) values
Movie=["Automata","Good People"]

Date=[1995,2015]

list_label=["Movie","Date"]

list_col=[Movie,Date]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
df["director"]=["henry","chriss"]

df["duration"]=0 #broadcasting

df
data.head()
data1.describe()
data["int_duration"]=data["duration"].str.split(n=1,expand=True)[0]

data["int_duration"]=data["int_duration"].astype(int)
data1=data.loc[:,["release_year","int_duration"]]

data1.plot()
data1.plot(subplots=True)
data1.plot(kind="scatter",x="release_year",y="int_duration")

plt.show()
data1.plot(kind = "hist",y = "release_year",bins = 20,range= (1950,2040),density = True)
data1.plot(kind = "hist",y = "release_year",bins = 20,range= (1950,2040),density = True,cumulative=True)

#density for normalized, cumulative to sum the previous ones

plt.savefig('graph.png')
data1.describe()
data=data.reset_index(drop=True)

data.head()
data["date_added"]
data["date_added"].value_counts(dropna=True)
# we'll convert date_added column to yyyy-mm-dd format

data["Month"]=data["date_added"].str.split(n=1,expand=True)[0]
data['Month']=["01" if i=="January" else "02" if i=="February" else "03" if i=="March" else "04" if i=="April" else "05" if i=="May" else "06" if i=="June" else "07" if i=="July" else "08" if i=="August" else "09" if i=="September" else "10" if i=="October" else "11" if i=="November" else "12" if i=="December" else "NaN" for i in data['Month']]
data['Month'].fillna("1", inplace = True) 

data["days"]=data["date_added"].str.split(n=1,expand=True)[1].str.split(',',n=1,expand=True)[0]
data["days"].fillna("1", inplace = True) 
data["year"]=data["date_added"].str.split(n=1,expand=True)[1].str.split(n=1,expand=True)[1]

data["year"].fillna("2000", inplace = True) 
data["date"]=data["year"]+"-"+data['Month']+"-"+data["days"]
data["date"]
del data["year"]

del data["Month"]

del data["days"]
#convert to datetime type

datetime_object=pd.to_datetime(data["date"])

datetime_object
dfm=data.copy()
dfm["date"]=datetime_object
dfm=dfm.set_index("date")  #time series

print(dfm.loc["2017-04-15"])
#if it was unique:

#dfm.loc["2019-08-30":"2017-04-15"]

#but this is non-unique
#A=year, M =month

dfm.resample("A").mean()
dfm.resample("M").mean()
dfm.resample("M").mean().interpolate("linear")  #Fills intervals as linear between upper and lower value.
data
#data["ind"]=0

#i=1

#while i<4266: 

#    data["ind"][i-1]=i 

#    i+=1

    

data["ind"]=range(1,4266,1)    
data.head()
data=data.set_index('ind')
data.head()
data["title"][1] #previous index was 0
data.title[1]
data.loc[1,"director"]
data[["date","title"]]
type(data["title"]) #series
type(data[["title"]]) #data frame
data.loc[1:10,"title":"cast"]
data.loc[10:1:-1,"title":"cast"]
data.loc[10:1:-1,"cast":]
data[data["release_year"]>2015]
first=data["release_year"]>2015 #first filter

second=data["type"]=="TV Show" #second fiter



data[first & second] #apply both
data.title[data.director=="Mariano Barroso"]
def inc(n):

    return n+1

data.show_id.apply(inc) #we can use "apply" for functions in data frames
data.show_id.apply(lambda n: n+1)

#data.show_id=...
# data["new_feature"]=data.int_duration+data.release_year 

#we can create a new feature using other columns.
print(data.index.name)
data.index.name="index"
data.head()
d=data.copy()

d.index=range(31,4296,1)

d.head()
d=d.set_index(["type","rating"])

d.head(10)
d.info()
d1=data.copy()  

d1=d1.loc[5:10,["release_year","duration","title"]]

d1.head()
d1.pivot(index="duration",columns="release_year",values="title")
d2=data.copy()

d2=d2.loc[60:145,["listed_in","date","title"]]

d2
d2 = d2.set_index(["title","listed_in"])

#d2 = d2.set_index(["title","listed_in"], append=True)

d2
# level determines indexes

d2.unstack(level=1) #if there are more than one index, this decrease it.
d2=d2.swaplevel(0,1)
d2
data.head()
pd.melt(data,id_vars="title",value_vars=["type","date"])
data.groupby("type").release_year.mean() 
data.groupby("int_duration").mean() 

#sum(),min(),max()..
data.groupby("type")[["release_year","int_duration"]].min() 