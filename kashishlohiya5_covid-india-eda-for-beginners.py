# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd

# data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

from datetime import datetime

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
data=pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")

data.head()
data['Date']=pd.to_datetime(data['Date'],dayfirst=True)
data['Month_Num']=pd.DatetimeIndex(data['Date']).month
data['Month']=0



data.loc[data['Month_Num']==1,'Month']='Jan'

data.loc[data['Month_Num']==2,'Month']='Feb'

data.loc[data['Month_Num']==3,'Month']='Mar'

data.loc[data['Month_Num']==4,'Month']='April'

data.loc[data['Month_Num']==5,'Month']='May'
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

print(dict_keys,'\n')



dict_values=dict.values()

print(dict_values)
plt.figure(figsize=(20,7))

ax=plt.bar(dict_keys,dict_values)

plt.xticks(rotation=90)

plt.title('Confirmed cases statewise',fontsize=30)

plt.xlabel('States',fontsize=30)

plt.ylabel('Number of Cases',fontsize=30)
#Displaying deaths

dict={}

for i in data_dict:

    data_loc=data[data['State/UnionTerritory'].str.contains(i)]

    #li.append(data_loc["Confirmed"].max())

    dict.update({i:data_loc["Deaths"].max()})

    

dict_keys=dict.keys()

print(dict_keys)



dict_values=dict.values()

print(dict_values)



plt.figure(figsize=(20,7))

ax=plt.bar(dict_keys,dict_values)

plt.xticks(rotation=90)



plt.title('Death cases statewise',fontsize=30)

plt.xlabel('States',fontsize=30)

plt.ylabel('Number of Deaths',fontsize=30)
fig = plt.figure(figsize=(10,10))

conf_per_state = data.groupby('State/UnionTerritory')['Confirmed'].max().sort_values(ascending=False)

#explode = conf_per_country

conf_per_state.plot(kind="pie",title='Percentage of confirmed cases per country',autopct='%1.1f%%', shadow= True)

data1=data.groupby('State/UnionTerritory')['Confirmed','Cured','Deaths'].max().sort_values('Confirmed',ascending=False)

data1=data1.reset_index()

data1
f, ax = plt.subplots(figsize=(10, 10))



bar1=sns.barplot(x="Confirmed",y="State/UnionTerritory",data=data1,

            label="Confirmed", color="b")





bar2=sns.barplot(x="Cured", y="State/UnionTerritory", data=data1,

            label="Cured", color="g")





bar3=sns.barplot(x="Deaths", y="State/UnionTerritory", data=data1,

            label="Deaths", color="r")



ax.legend(loc=4, ncol = 1)

plt.show()


data1["Recovery Rate"]=data1['Cured']/data1["Confirmed"]

data1['Death Rate']=data1['Deaths']/data1['Confirmed']
data1


print('Recovery Rate=',data1['Recovery Rate'].mean()*100,"%")

print('Death Rate=',data1['Death Rate'].mean()*100,"%")
f,ax= plt.subplots(figsize=(15,10))

#fig, ax = plt.subplots(6, 6,figsize=(15,5))

#for i in range(6):



#for j in range(6):

#for i in dict:

data_state=data[data['State/UnionTerritory'].str.contains('Rajasthan')]

data_state["Recovery Rate"]=data_state['Cured']/data_state["Confirmed"]

data_state['Death Rate']=data_state['Deaths']/data_state['Confirmed']

   

plt.plot(data_state['Date'],data_state['Recovery Rate'], marker='o',label="Recovery Rate")

plt.plot(data_state['Date'],data_state['Death Rate'], marker='*',label="Death Rate")

plt.ylabel('Rate',fontsize=20)

plt.xlabel('Date',fontsize=20)

plt.xticks(rotation=90)

plt.title('Recovery and Death Rate graph of state Rajasthan',fontsize=20)

ax.legend(loc='upper left',fontsize=20)



#data_state.tail()
f,ax= plt.subplots(figsize=(15,10))

#fig, ax = plt.subplots(6, 6,figsize=(15,5))

#for i in range(6):



#for j in range(6):

#for i in dict:

#data_state=data[data['State/UnionTerritory'].str.contains('Rajasthan')]

#data_state["Recovery Rate"]=data_state['Cured']/data_state["Confirmed"]

#data_state['Death Rate']=data_state['Deaths']/data_state['Confirmed']

   

plt.plot(data_state['Date'],data_state['Confirmed'], marker='o',label="Confirmed")

plt.plot(data_state['Date'],data_state['Deaths'], marker='*',label="Deaths")

plt.plot(data_state['Date'],data_state['Cured'], marker='v',label="Cured")



plt.ylabel('Rate',fontsize=30)

plt.xlabel('Date',fontsize=30)

plt.xticks(rotation=90)

plt.title('Confirmed,Deaths and Cured  graph of state Rajasthan',fontsize=20)

ax.legend(loc='upper left',fontsize=20)