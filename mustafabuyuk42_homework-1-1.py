# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import  matplotlib.pyplot as plt

import seaborn as sns #Visualition tool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/master.csv')
data.info()
data.corr
#correlation map

f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.2f',ax=ax)
data.head
data.columns
data.year.plot(kind='line',color='g',label='year',linewidth=1,alpha=.9,grid=True,linestyle=':')

data.suicides_no.plot(color='r',label='suicides_no',linewidth=1,alpha=.5,grid=True,linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('suicides_no')

plt.ylabel('year')

plt.title('line plot')

plt.show()
data.plot(kind='scatter',y='suicides_no',x='year',color='r',alpha=.5)

data.suicides_no.plot(kind='hist',bins=10,figsize=(18,18))

plt.show()
data.tail()
data=pd.read_csv('../input/master.csv')

data.head(8)

data.columns
data.shape
data.info
data.columns
v=data['age'].value_counts(dropna=False)

print(v)
print(data['generation'].value_counts(dropna=False))
data.describe()
data.boxplot(column='age',by='suicides_no')
data_1=data.head()

data_1
melted=pd.melt(frame=data_1,id_vars='sex',value_vars=['suicides_no','gdp_per_capita ($)'])

melted
data.columns
data1=data['sex'].head()

data2=data['suicides_no'].head()

con_con=pd.concat([data1,data2],axis=1)

con_con
data1=data.head()

data2=data.tail()

concat1=pd.concat([data1,data2],axis=0,ignore_index=True)

concat1
data.dtypes
data['population']=data['population'].astype('float32')

data['country']=data['country'].astype('category')

data.dtypes
data['HDI for year'].value_counts(dropna='False')
assert data['HDI for year'].notnull().all()
assert data.age.dtypes==np.int
country=['turkey','france']

population=['10','5']

list_label=['country','population']

list_col=[country,population]

zipped=zip(list_label,list_col)

zipped=list(zipped)

zipped

data_dict=dict(zipped)

data_dict

df=pd.DataFrame(data_dict)

df
df['capital']=['ankara','paris']

df
data.columns
data1=data.loc[:,['suicides/100k pop','gdp_per_capita ($)','population']]

data1.plot()
data1.plot(subplots='True')

plt.show()
data1.plot(kind='scatter',y='suicides/100k pop',x='gdp_per_capita ($)')

plt.show()
data1.plot(kind='hist',y='suicides/100k pop',bins=50,range=(0,250),normed=True)

plt.show()
time_list=["1994-1-10","1994-2-11","1994-3-12"]

print(type(time_list))

datetime_object=pd.to_datetime(time_list)

print(type(datetime_object))
data2=data.head()

time_list=["1994-1-10","1994-1-9","1994-3-10","1994-4-10","1994-5-10"]

time_obj=pd.to_datetime(time_list)

data2["date"]=time_obj

data2=data2.set_index("date")

data2
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")

data=pd.read_csv('../input/master.csv')

#data=data.set_index('country')

data.head()
data[["sex","age","population"]]
print(type(data["age"]))

print(type(data[["age"]]))
data.loc[1:10,"year":"suicides_no"]
boolean=data.suicides_no>100

data[boolean]
first_filter=data.population>100000

sec_filter=data.year>2000

third_filter=data.suicides_no<10

data[first_filter & sec_filter & third_filter]
print(data.index.name)
data.index.name="index name"

data.head()
data3=data.copy()

data3.index=range(100,27920,1)

data3.head()
data=pd.read_csv('../input/master.csv')

data.head()
data1=data.set_index(["country","sex"])

data1.head(1000)
dict1={"treatment":["A","A","B","B"],"response":[1,2,9,10],"gender":["M","F","M","M"],"age":[10,20,15,30]}

df=pd.DataFrame(dict1)

df
# pivoting

df.pivot(index="treatment",columns = "gender",values="age")
df1=df.set_index(["treatment","gender"])

df1
df1.unstack(level=0)
df1.unstack(level=1)
df
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
pd.melt(data.head(10),id_vars="country",value_vars=["age","sex"])
df.groupby("response").mean()
df.groupby("treatment").age.max()
df.groupby("treatment")[["age","response"]].min() 
df.groupby("gender")[["age","response"]].min() 
data.head()
data.groupby("year")[["population","suicides_no","suicides/100k pop"]].mean()