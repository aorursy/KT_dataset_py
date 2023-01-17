# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/merged.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/covid19/merged.csv')

data.head
# Importing the libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Data Processing

data['Active'] = data['Confirmed'] - (data['Recovered'] + data['Deaths'])

data['Date'] =  pd.to_datetime(data['Date'], format='%m/%d/%Y')

data['Date1'] = pd.to_datetime('2020/01/22')

data['Days']= data['Date'] - data['Date1']

data['Days'] = data['Days'].astype(str).str[:2]

data['Days'] = data[['Days']].apply(pd.to_numeric)

data.drop('Date1',inplace=True, axis=1)
# Country wise confirmed and recovered cases

temp=data.loc[data['Date'] == '3/19/2020']

temp=temp.loc[data['Confirmed'] > 500]

temp.drop('Province/State', axis=1, inplace=True)

temp = temp.groupby('Country/Region').agg({'Confirmed': 'sum', 'Recovered': 'sum', 'Deaths': 'sum', 'Active': 'sum' }).reset_index()

temp=temp.sort_values(by=['Confirmed'],ascending=False).reset_index()

country_wise=temp.drop(['index'], axis=1)

country_wise
# Graph showing the global confirmed cases

t1 = temp.melt(id_vars=['Country/Region'], value_vars=['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Cases', value_name='Count')

t1 = t1.groupby(['Country/Region','Cases'])['Count'].sum().reset_index()

sns.set(style="darkgrid")

sns.catplot(x="Country/Region", y="Count", hue="Cases", data=t1, height=5, aspect=2, kind='bar')

plt.title('People infected globally')

plt.xlabel('Count of cases')

plt.ylabel('Name of the country')

plt.xticks(rotation=90)

plt.show()
# Graph showing the number of cases in china

temp=data.loc[data['Country/Region'] == 'China']

temp=temp.loc[data['Date'] == '3/19/2020']

t2 = temp.melt(id_vars=['Province/State'], value_vars=['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Cases', value_name='Count')

t2 = t2.groupby(['Province/State','Cases'])['Count'].sum().reset_index()

sns.set(style="darkgrid")

sns.catplot(x="Province/State", y="Count", hue="Cases", data=t2, height=5, aspect=2, kind='bar')

plt.title('Comparision of Cases in different provinces of China')

plt.ylabel('Count of cases')

plt.xlabel('Name of the country')

plt.xticks(rotation=90)

plt.show()
# Graph showing cases outside china

temp=data.loc[data['Country/Region'] != 'China']

temp=temp.loc[data['Date'] == '3/19/2020']

temp = temp.groupby('Country/Region').agg({'Confirmed': 'sum', 'Recovered': 'sum', 'Deaths': 'sum', 'Active': 'sum' }).reset_index()

temp=temp.sort_values(by=['Confirmed'],ascending=False).reset_index()

temp.drop(['index'], axis=1, inplace=True)

temp = temp.head(15)

t2 = temp.melt(id_vars=['Country/Region'], value_vars=['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Cases', value_name='Count')

t2 = t2.groupby(['Country/Region','Cases'])['Count'].sum().reset_index()

sns.set(style="darkgrid")

sns.catplot(x="Country/Region", y="Count", hue="Cases", data=t2, height=5, aspect=2, kind='bar')

plt.title('Comparision of Cases outside China')

plt.xlabel('Count of cases')

plt.ylabel('Name of the country')

plt.xticks(rotation=90)

plt.show()
# Timeline of the spread of the disease Globally

t2 = data.melt(id_vars=['Days'], value_vars=['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Cases', value_name='Count')

t2 = t2.groupby(['Days','Cases'])['Count'].sum().reset_index()

t2=t2.sort_values(by=['Days'],ascending=True).reset_index()

t2.drop(['index'], axis=1, inplace=True)

plt.figure(figsize=(12,6))

sns.lineplot(x="Days", y="Count", hue="Cases", data=t2, style='Cases', markers=True)

plt.title('Timeline of the spread of the disease Globally')

plt.xlabel('Days')

plt.ylabel('Number of Cases')

plt.xticks(rotation=90)

plt.show()

# Spread of cases in Hubei and outside Hubei

temp=data.loc[data['Country/Region'] == 'China']

t1 = temp.loc[data['Province/State'] == 'Hubei']

t2 = temp.loc[data['Province/State'] != 'Hubei']

fig, axs = plt.subplots(ncols=2, figsize=(16,6))

t1 = t1.melt(id_vars=['Days'], value_vars=['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Cases', value_name='Count')

t1 = t1.groupby(['Days','Cases'])['Count'].sum().reset_index()

t1=t1.sort_values(by=['Days'],ascending=True).reset_index()

t1.drop(['index'], axis=1, inplace=True)

ax=axs[0].set_title('Hubei Province').set_fontsize('18')

sns.lineplot(x="Days", y="Count", hue="Cases", data=t1, style='Cases', markers=True,  ax=axs[0])

t2 = t2.melt(id_vars=['Days'], value_vars=['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Cases', value_name='Count')

t2 = t2.groupby(['Days','Cases'])['Count'].sum().reset_index()

t2=t2.sort_values(by=['Days'],ascending=True).reset_index()

t2.drop(['index'], axis=1, inplace=True)

sns.lineplot(x="Days", y="Count", hue="Cases", data=t2, style='Cases', markers=True,  ax=axs[1])

ax=axs[1].set_title('Non-Hubei Province').set_fontsize('18')

plt.show()
# Comparision of spread of the disease in China and Globally

temp=data.loc[data['Country/Region'] == 'China']

fig, axs = plt.subplots(ncols=2, figsize=(16,6))

t1 = data.melt(id_vars=['Days'], value_vars=['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Cases', value_name='Count')

t1 = t1.groupby(['Days','Cases'])['Count'].sum().reset_index()

t1=t1.sort_values(by=['Days'],ascending=True).reset_index()

t1.drop(['index'], axis=1, inplace=True)

ax=axs[0].set_title('Global Trends').set_fontsize('18')

sns.lineplot(x="Days", y="Count", hue="Cases", data=t1, style='Cases', markers=True,  ax=axs[0])

t2 = temp.melt(id_vars=['Days'], value_vars=['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Cases', value_name='Count')

t2 = t2.groupby(['Days','Cases'])['Count'].sum().reset_index()

t2=t2.sort_values(by=['Days'],ascending=True).reset_index()

t2.drop(['index'], axis=1, inplace=True)

sns.lineplot(x="Days", y="Count", hue="Cases", data=t2, style='Cases', markers=True,  ax=axs[1])

ax=axs[1].set_title('Trends in China').set_fontsize('18')

plt.show()
# Spread of the disease in different parts of the world

country_list=['China','Italy','Iran','Korea, South','Spain','Germany']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15,10))

m=0

l=0

for i in country_list:

    temp=data.loc[data['Country/Region'] == i]

    t2 = temp.melt(id_vars=['Days'], value_vars=['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Cases', value_name='Count')

    t2 = t2.groupby(['Days','Cases'])['Count'].sum().reset_index()

    t2=t2.sort_values(by=['Days'],ascending=True).reset_index()

    t2.drop(['index'], axis=1, inplace=True)

    if l == 3 and m==0:

        m+=1

        l = 0

    sns.lineplot(x="Days", y="Count", hue="Cases", data=t2, style='Cases', markers=True, ax=axs[m,l])

    ax=axs[m,l].set_title(i).set_fontsize('16')

    l+=1
temp=data.loc[data['Date'] == '3/19/2020']

temp = temp.groupby('Country/Region')['Confirmed'].sum().reset_index()

temp=temp.sort_values(by=['Confirmed'],ascending=False)

t1=data.loc[data['Date'] == '3/17/2020']

t1 = t1.groupby('Country/Region')['Confirmed'].sum().reset_index()

t1=t1.sort_values(by=['Confirmed'],ascending=False)

fresh_cases=pd.merge(temp,t1,on='Country/Region')

fresh_cases['Cases']=fresh_cases['Confirmed_x'] - fresh_cases['Confirmed_y']

fresh_cases=fresh_cases.head(20)

plt.figure(figsize=(8,4))

sns.barplot(y='Country/Region', x='Cases', data=fresh_cases)

plt.title('Number of new cases reported globally')

plt.xlabel('Number of cases')

plt.ylabel('Name of the country')
country_wise['death_ratio']=country_wise['Deaths']/country_wise['Confirmed']*100

country_wise=country_wise.sort_values(by=['death_ratio'],ascending=False)

country_wise =country_wise.head(20)

plt.figure(figsize=(8,4))

sns.barplot(y='Country/Region', x='death_ratio', data=country_wise)

plt.title('Death ratio in different countries')

plt.xlabel('Ratio')

plt.ylabel('Name of the country')
# Spread of cases in India

temp=data.loc[data['Country/Region'] == 'China']

t3=data.loc[data['Country/Region'] == 'India']

fig, axs = plt.subplots(ncols=2, figsize=(16,6))

t1 = temp.melt(id_vars=['Days'], value_vars=['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Cases', value_name='Count')

t1 = t1.groupby(['Days','Cases'])['Count'].sum().reset_index()

t1=t1.sort_values(by=['Days'],ascending=True).reset_index()

t1.drop(['index'], axis=1, inplace=True)

ax=axs[0].set_title('Spread in China').set_fontsize('18')

sns.lineplot(x="Days", y="Count", hue="Cases", data=t1, style='Cases', markers=True,  ax=axs[0])

t2 = t3.melt(id_vars=['Days'], value_vars=['Confirmed', 'Recovered', 'Deaths', 'Active'], var_name='Cases', value_name='Count')

t2 = t2.groupby(['Days','Cases'])['Count'].sum().reset_index()

t2=t2.sort_values(by=['Days'],ascending=True).reset_index()

t2.drop(['index'], axis=1, inplace=True)

sns.lineplot(x="Days", y="Count", hue="Cases", data=t2, style='Cases', markers=True,  ax=axs[1])

ax=axs[1].set_title('Spread in India').set_fontsize('18')

plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/complete.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
India=pd.read_csv('/kaggle/input/covid19/complete.csv')

India.head()
India['Date'] =  pd.to_datetime(India['Date'], format='%m/%d/%Y')

India['Date1'] = pd.to_datetime('2020/01/30')

India['Days']= India['Date'] - India['Date1']

India['Days'] = India['Days'].astype(str).str[:2]

India['Days'] = India[['Days']].apply(pd.to_numeric)

India.drop(['Date1','Cured/Discharged/Migrated'],inplace=True, axis=1)

India.head()
temp1=India.loc[India['Date'] == '3/19/2020']

temp1.drop(['Total Confirmed cases (Indian National)',

           'Total Confirmed cases ( Foreign National )',

           'Latitude',

           'Longitude',

           'Death',

           'Recovered',

           'Active'], axis=1, inplace=True)

temp1=temp1.sort_values(by=['Confirmed'],ascending=False)

plt.figure(figsize=(8,4))

sns.barplot(y='Name of State / UT', x='Confirmed', data=temp1)

plt.title('Number of confirmed cases in India')

plt.xlabel('Number of confirmed cases in each state')

plt.ylabel('Name of the state/UT')

plt.show()
# Timeline of the spread of the diseasein India

t2 = India.melt(id_vars=['Days'], value_vars=['Confirmed', 'Recovered', 'Death', 'Active'], var_name='Cases', value_name='Count')

t2 = t2.groupby(['Days','Cases'])['Count'].sum().reset_index()

t2=t2.sort_values(by=['Days'],ascending=True).reset_index()

t2.drop(['index'], axis=1, inplace=True)

plt.figure(figsize=(12,6))

sns.lineplot(x="Days", y="Count", hue="Cases", data=t2, style='Cases', markers=True)

plt.title('Timeline of the spread of the disease in India')

plt.xlabel('Days')

plt.ylabel('Number of Cases')

plt.xticks(rotation=90)

plt.show()
# Spread of the disease in different parts of the India

state_list=[  'Delhi',

              'Haryana',

              'Karnataka',

              'Kerala',

              'Maharashtra',

              'Rajasthan',

              'Telengana',

              'Union Territory of Jammu and Kashmir',

              'Union Territory of Ladakh',

              'Uttar Pradesh']

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20,15))

m=0

l=0

for i in state_list:

    temp=India.loc[India['Name of State / UT'] == i]

    t2 = temp.melt(id_vars=['Days'], value_vars=['Confirmed', 'Recovered', 'Death', 'Active'], var_name='Cases', value_name='Count')

    t2 = t2.groupby(['Days','Cases'])['Count'].sum().reset_index()

    t2=t2.sort_values(by=['Days'],ascending=True).reset_index()

    t2.drop(['index'], axis=1, inplace=True)

    if l == 5 and m==0:

        m+=1

        l = 0

    sns.lineplot(x="Days", y="Count", hue="Cases", data=t2, style='Cases', markers=True, ax=axs[m,l])

    ax=axs[m,l].set_title(i).set_fontsize('18')

    l+=1