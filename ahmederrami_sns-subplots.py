# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#path of the file to read
path="../input/corona-virus-report/covid_19_clean_complete.csv"

#read the file into a variable df
df = pd.read_csv(path,index_col='Date',parse_dates=True)

#remove "Province/State" column
df.drop(['Province/State'], axis=1, inplace=True)
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
#our analysis will focus on some countries we put in a list
countries = ["Morocco","Algeria","Egypt","Nigeria","Iraq","Ghana"]
#countries = ["Morocco","Algeria","Italy"]

#from the dataset, we will extract the data concerning the countries in the list. we are interested in the period after March 1st
data=df[(df["Country/Region"].isin(countries)) & (df.index>='2020-02-15')]
fig=plt.figure(figsize=(16,8))

#line chart showing the day-to-day accumulation of confirmed cases for each country in the list
sns.lineplot(x=data.index,y=data['Confirmed'],hue="Country/Region", data=data)
fig = plt.figure(figsize=(16,15))
ax = fig.subplots(3,sharex=True)

sns.lineplot(x=data.index,y=data['Confirmed'],hue="Country/Region", data=data, ax=ax[0])
ax[0].set_title("Total confirmed cases",fontsize=16, color="red")
sns.lineplot(x=data.index,y=data['Recovered'],hue="Country/Region", data=data, ax=ax[1])
ax[1].set_title("Total recovered",fontsize=16, color="red")
sns.lineplot(x=data.index,y=data['Deaths'],hue="Country/Region", data=data, ax=ax[2])
ax[2].set_title("Total deaths",fontsize=16, color="red")
data['pDeaths']=data['Deaths']/(data['Deaths']+data['Recovered'])
data['pRecovered']=data['Recovered']/(data['Deaths']+data['Recovered'])

data.dropna(inplace=True)
fig = plt.figure(figsize=(16,5*len(countries)))
ax = fig.subplots(len(countries),sharex=True)
i=0
for country in countries:
    sns.lineplot(data=data[data["Country/Region"]==country][["pDeaths","pRecovered"]], ax=ax[i])
    ax[i].set_title(country, fontsize=16, color="red")
    i=i+1

data['activeCases']=data['Confirmed']-data['Deaths']-data['Recovered']
fig = plt.figure(figsize=(16,10))
ax = fig.subplots()

sns.lineplot(x=data.index,y=data['activeCases'],hue="Country/Region", data=data)
ax.set_title("Total active cases",fontsize=16, color="red")
fig=plt.figure(figsize=(16,4*len(countries)))
ax=fig.subplots(len(countries),sharey=True)
i=0
for country in countries:
    _data = data[data["Country/Region"]==country]
    sns.barplot(x=_data.index, y=_data["activeCases"], ax=ax[i])
    ax[i].set_title(country, color="red", fontsize=16)
    ax[i].set_xlabel("")
    ax[i].set_xticks([])
    i=i+1
