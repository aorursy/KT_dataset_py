# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/covid_19_india.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#First we will work on Main Dataset - Covid_19_data
#we'll explore, analyise, visualize and predict for main dataset



##df_india = df_covid19[df_covid19['Country/Region']=='India']

df_india = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
df_india.head()
df_india.info()
#let's drop the column Sno and rename the columns 
df_india.drop('Sno',axis=1,inplace=True)
df_india.rename(columns={'Cured': 'Recovered','State/UnionTerritory':'State','ConfirmedIndianNational':'IndianNational','ConfirmedForeignNational':'Foreigners'},inplace=True)
df_india.info()
#let's see if we have some missing values
df_india.isna().sum()
#Visualization of Deaths over the time
plt.figure(figsize=(15,5))
sns.barplot(x=df_india['Date'],y=df_india['Deaths'])
plt.xticks(rotation=90)
#Visualization of Confirmed cases over the time
plt.figure(figsize=(15,5))
sns.barplot(x=df_india['Date'],y=df_india['Confirmed'])
plt.xticks(rotation=90)
#Visualization of Recovred cases over the time
plt.figure(figsize=(15,5))
sns.barplot(x=df_india['Date'],y=df_india['Recovered'])
plt.xticks(rotation=90)
#Joint Plot between Recovered and Death cases
sns.jointplot(x='Recovered',y='Deaths',data=df_india)
#Joint Plot between Confirmed and Death cases
sns.jointplot(x='Confirmed',y='Deaths',data=df_india)

#State wise deaths Cases
plt.figure(figsize=(15,5))
df_state = df_india.groupby(by=['State'],as_index=False)['Confirmed','Deaths','Recovered'].max().reset_index()
df_state.drop('index',axis=1,inplace=True)
print(df_state['Deaths'].sum())
print(df_state['Confirmed'].sum())
print(df_state['Recovered'].sum())
sns.barplot(x=df_state['State'],y=df_state['Deaths'])
plt.xticks(rotation=90)
#State wise Confirmed Cases
plt.figure(figsize=(15,5))
sns.barplot(x=df_state['State'],y=df_state['Confirmed'])
plt.xticks(rotation=90)
#State wise recovered cases
plt.figure(figsize=(15,5))
sns.barplot(x=df_state['State'],y=df_state['Recovered'])
plt.xticks(rotation=90)
