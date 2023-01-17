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

        

import datetime as dt



#for visualizatin

import matplotlib.pyplot as plt

import seaborn as sns



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/covid19-india-prediction/covid-19-india.csv')
data
#renaming column names

data.columns=['State', 'Total_Confirmed_cases',

       'Cured_Discharged_Migrated', 'Death', 'Date', 'Latitude', 'Longitude',

       'Total_cases']
data.head()
data.tail()
data.info()
data.columns
data.describe(include='all')
#check the shape of this datase

data.shape
#check the datatypes of this dataset

data.dtypes
#check the null values

data.isnull().sum()
#clean the nan value

data['Latitude'].fillna(data['Latitude'].mean(), inplace=True)

data['Longitude'].fillna(data['Longitude'].mean(), inplace=True)
#after cleaning the nan value check the null value

data.isnull().sum()
#current date

today = data[data.Date == '2020-05-29']
today
max_Cured_Discharged_Migrated_cases=today.sort_values(by="Cured_Discharged_Migrated",ascending=False)

max_Cured_Discharged_Migrated_cases
#Getting states with maximum number of confirmed cases

top_states_confirmed=max_Cured_Discharged_Migrated_cases[0:5]

top_states_confirmed
max_death_cases=today.sort_values(by="Death",ascending=False)

max_death_cases
#Sorting data w.r.t number of death cases

max_death_cases=today.sort_values(by="Death",ascending=False)

max_death_cases
top_state_death=max_death_cases[0:5]

top_state_death
data
#Making bar-plot for states with top confirmed cases

sns.set(rc={'figure.figsize':(15,10)})

sns.barplot(x="State",y="Total_Confirmed_cases",data=top_states_confirmed,hue="State")

plt.show()
#Making bar-plot for states with Cured_Discharged_Migrated

sns.set(rc={'figure.figsize':(15,10)})

sns.barplot(x="State",y="Cured_Discharged_Migrated",data=top_states_confirmed,hue="State")

plt.show()
#Making bar-plot for states with death

sns.set(rc={'figure.figsize':(15,10)})

sns.barplot(x="State",y="Death",data=top_states_confirmed,hue="State")

plt.show()
#Making bar-plot for Total_cases with date and top_states_confirmed

sns.set(rc={'figure.figsize':(15,10)})

sns.barplot(x="Total_cases",y="Date",data=top_states_confirmed,hue="State")

plt.show()
sns.catplot(y='State',x='Total_Confirmed_cases',kind='bar',data=data)

plt.show()
sns.catplot(y='State',x='Death',kind='bar',data=data)

plt.show()
#Making bar-plot for states with total_cases

sns.set(rc={'figure.figsize':(15,10)})

sns.barplot(x="State",y="Total_cases",data=top_states_confirmed,hue="State")

plt.show()
#Visualizing in Latitude and death 

sns.set(rc={'figure.figsize':(15,10)})

sns.lineplot(x="Latitude",y="Death",data=data,color="g")

plt.show()
#Visualizing in Longitude and death 

sns.set(rc={'figure.figsize':(15,10)})

sns.lineplot(x="Longitude",y="Death",data=data,color="r")

plt.show()