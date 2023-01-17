# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns

import re



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")
data.sample(10)
data.isnull().sum()
data['State/UnionTerritory'].value_counts()
import re

data=data.replace(to_replace='\#',value='', regex=True)

data['State/UnionTerritory'].value_counts()
data_dict=data['State/UnionTerritory'].value_counts().to_dict()

data_dict


dict={}

for i in data_dict:

    data_loc=data[data['State/UnionTerritory'].str.contains(i)]

    #li.append(data_loc["Confirmed"].max())

    dict.update({i:data_loc["Confirmed"].max()})

    

dict
dict_keys=dict.keys()

print(dict_keys)



dict_values=dict.values()

print(dict_values)
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(20,7))

ax=plt.bar(dict_keys,dict_values)

plt.xticks(rotation=90)



#for i in dict_keys,dict_values:

    

   # plt.annotate(xy=(dict_keys[i],dict_values[i]))
dict={}

for i in data_dict:

    data_loc=data[data['State/UnionTerritory'].str.contains(i)]

    #li.append(data_loc["Confirmed"].max())

    dict.update({i:data_loc["Deaths"].max()})

    

dict
dict_keys=dict.keys()

print(dict_keys)



dict_values=dict.values()

print(dict_values)



plt.figure(figsize=(20,7))

ax=plt.bar(dict_keys,dict_values)

plt.xticks(rotation=90)

fig = plt.figure(figsize=(10,10))

conf_per_state = data.groupby('State/UnionTerritory')['Confirmed'].sum().sort_values(ascending=False)

#explode = conf_per_country

conf_per_state.plot(kind="pie",title='Percentage of confirmed cases per country',autopct='%1.1f%%', shadow= True)



group_cases=data[['Confirmed','Cured','Deaths','State/UnionTerritory']].groupby('State/UnionTerritory').max().sort_values('Confirmed',ascending=False).head(10)

group_cases=group_cases.reset_index()

print(group_cases.shape)

group_cases
f, ax = plt.subplots(figsize=(15, 10))



bar1=sns.barplot(x="Confirmed",y="State/UnionTerritory",data=group_cases,

            label="Confirmed", color="b")





bar2=sns.barplot(x="Cured", y="State/UnionTerritory", data=group_cases,

            label="Cured", color="g")





bar3=sns.barplot(x="Deaths", y="State/UnionTerritory", data=group_cases,

            label="Deaths", color="r")



ax.legend(loc=4, ncol = 1)

plt.show()
data_table= data[["Cured","Deaths","Confirmed","State/UnionTerritory"]].groupby("State/UnionTerritory").max().sort_values('Confirmed',ascending=False)

data_table=data_table.reset_index()

data_table.shape

data_table
data_table["Recovery Rate"]=data_table['Cured']/data_table["Confirmed"]

data_table['Death Rate']=data_table['Deaths']/data_table['Confirmed']
data_table
data_state=data[data['State/UnionTerritory'].str.contains('Rajasthan')]

data_state["Recovery Rate"]=data_state['Cured']/data_state["Confirmed"]

data_state['Death Rate']=data_state['Deaths']/data_state['Confirmed']



data_state.tail()
#sns.lineplot(data_state['Date'],data_state['Recovery Rate'])

f = plt.subplots(figsize=(15,5))

plt.plot(data_state['Date'][:-20],data_state['Recovery Rate'][:-20], marker='o')

plt.plot(data_state['Date'][:-20],data_state['Death Rate'][:-20], marker='*')



plt.xticks(rotation=90)





#data_state['Recovery Rate'].plot(figsize=(15,8))

#data_state['Death Rate'].plot(figsize=(15,8))

data_state
fig = plt.figure(figsize=(10,10))

rr_per_state = data_state.groupby('State/UnionTerritory')['Recovery Rate'].sum().sort_values(ascending=False)

#explode = conf_per_country

rr_per_state.plot(kind="pie",title='Percentage of confirmed cases per country',autopct='%1.1f%%', shadow= True)
from datetime import datetime 

import datetime as dt

#date= datetime.now()
data_state['month']=pd.to_datetime(data_state['Date'],dayfirst=True)



#data_date=pd.to_datetime(data_state['Date'],format='%d%m%Y',errors='ignore')

#data_date.shape

#my_date = datetime.strptime(data_state['Date'], "%Y-%m-%d")
data_state['monthly']= pd.DatetimeIndex(data_state['month']).month



#data_state.head(15)

data_state.sample(10)
data_state['monthly'].value_counts()
data_state['mon']=0

data_state.loc[data_state['monthly']==5,'mon']='May'

data_state.loc[data_state['monthly']==4,'mon']='April'

data_state.loc[data_state['monthly']==3,'mon']='March'



data_state['mon'].value_counts()

data_state.sample(10)
data_state.head()

data_state.drop(['Date','monthly'], axis=1,inplace=True)
data_state.head(10)
data_state_march=data_state.loc[data_state['mon']=="May"]

data_state_march

f = plt.subplots(figsize=(10,5))

plt.plot(data_state_march['month'],data_state_march['Recovery Rate'], marker='o')

plt.plot(data_state_march['month'],data_state_march['Death Rate'], marker='*')





#plt.plot(data_state['Date'][:-20],data_state['Recovery Rate'][:-20], marker='o')

#plt.plot(data_state['Date'][:-20],data_state['Death Rate'][:-20], marker='*')



plt.xticks(rotation=90)

grp.head()
data.tail()

#data.groupby("State/UnionTerritory")["Confirmed"].sum().sort_values()