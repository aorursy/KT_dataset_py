import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

sns.set_style("darkgrid")



import warnings

#Suppressing all warnings

warnings.filterwarnings("ignore")



%matplotlib inline



df = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')

df.drop([df.columns[0], df.columns[1]], axis=1, inplace=True)

df['Country'] = df['Location'].apply(lambda location: location.split(',')[-1])

df['Day']=df['Datum'].apply(lambda datum: datum.split()[0])

df['Month']=df['Datum'].apply(lambda datum: datum.split()[1])

df['Date']=df['Datum'].apply(lambda datum: datum.split()[2][:2]).astype(int)

df['Year']=df['Datum'].apply(lambda datum: datum.split()[3]).astype(int)

df['Hour']=df['Datum'].apply(lambda datum: int(datum.split()[-2][:2]) if datum.split()[-1]=='UTC' else np.nan)

df.rename(columns={df.columns[5]: 'Rocket'}, inplace=True)

df['Rocket'] = df['Rocket'].fillna(0.0).str.replace(',', '')

df.Rocket = df.Rocket.apply(lambda x: str(x).strip()).astype('float64')

df.drop(['Datum', 'Location', 'Detail'], 1, inplace=True)

df.head()
df.describe(include='all')
# Combining small records into a single variable "Other"

comp_count = df['Company Name'].value_counts()

other = 0

ind = []

for i in comp_count.index:

    if comp_count[i]<5:

        other+=comp_count[i]

        ind.append(i)



#Creating Dataframe for plotting

companies = df['Company Name']

companies.replace(ind, 'Other', inplace=True)

companies_df=pd.DataFrame()

companies_df['Companies']=companies

comp_count = companies_df['Companies'].value_counts()



#Order of plotting

order = list(comp_count.index.values)

order.remove('Other')

order.append('Other')



#CountPlot

fig, ax = plt.subplots(figsize=(20, 15))

ax.set_title('No. of Launches by Company', fontsize=20)

sns.countplot(y='Companies', data=companies_df, order=order)

ax.set_xlabel('Companies', fontsize=15)

ax.set_ylabel('No. of Launches', fontsize=15)

plt.show()
companies_df=companies_df[companies_df['Companies']!='RVSN USSR']

order.remove('RVSN USSR')

fig, ax = plt.subplots(figsize=(20, 15))

ax.set_title('No. of Launches by Company (Excluding RVSN USSR)', fontsize=20)

sns.countplot(y='Companies', data=companies_df, order=order)

ax.set_xlabel('Companies', fontsize=15)

ax.set_ylabel('No. of Launches', fontsize=15)

plt.show()
df['Status Rocket'].replace(['StatusRetired', 'StatusActive'],['Retired', 'Active'], inplace=True)

fig = px.pie(df,names='Status Rocket', title='Status of Rocket',width=600, height=400)

fig.show()
fig = px.pie(df,names='Status Mission', title='Status of Mission',width=600, height=400)

fig.show()
fig, ax = plt.subplots(figsize=(20, 10))

ax.set_title('No. of Launches by Country', fontsize=20)

sns.countplot(y='Country', data=df, order=df['Country'].value_counts().index)

ax.set_xlabel('Countries', fontsize=15)

ax.set_ylabel('No. of Launches', fontsize=15)

plt.show()
data = df.groupby(['Company Name'])['Rocket'].sum().reset_index()

data = data[data['Rocket'] > 0].sort_values(by='Rocket', ascending=False)

data.columns = ['Company', 'Money']

fig = px.bar(

    data, 

    x='Company', 

    y="Money", 

    orientation='v', 

    title='Total Money spent on Missions by Company', 

    width=800,

    height=600

)

fig.show()
data = df.groupby(['Country'])['Rocket'].sum().reset_index()

data = data[data['Rocket'] > 0].sort_values(by='Rocket', ascending=False)

data.columns = ['Country', 'Money']

fig = px.bar(

    data, 

    x='Country', 

    y="Money", 

    orientation='v', 

    title='Total Money spent on Missions by Country', 

    width=800,

    height=600

)

fig.show()
fig, ax = plt.subplots(figsize=(10, 10))

ax.set_title('No. of Launches by Day of the Week', fontsize=20)

day_df=pd.DataFrame()

day_df['Day']=df['Day'].value_counts().index

day_df['Launches']=df['Day'].value_counts().values

order = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

sorter = [order.index(i) for i in day_df['Day']]

day_df['Sorter']=sorter

sns.lineplot(x = 'Day', y = "Launches", data=day_df.sort_values(by='Sorter'), sort=False)

ax.set_xlabel('Day', fontsize=15)

ax.set_ylabel('No. of Launches', fontsize=15)

plt.show()
fig, ax = plt.subplots(figsize=(10, 10))

ax.set_title('No. of Launches by Month', fontsize=20)

month_df=pd.DataFrame()

month_df['Month']=df['Month'].value_counts().index

month_df['Launches']=df['Month'].value_counts().values

order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sorter = [order.index(i) for i in month_df['Month']]

month_df['Sorter']=sorter

sns.lineplot(x = 'Month', y = "Launches", data=month_df.sort_values(by='Sorter'), sort=False)

ax.set_xlabel('Month', fontsize=15)

ax.set_ylabel('No. of Launches', fontsize=15)

plt.show()
fig, ax = plt.subplots(figsize=(20, 10))

ax.set_title('No. of Launches by Year', fontsize=20)

sns.countplot(x='Year', data=df, order=df['Year'].unique().sort())

ax.set_xlabel('Year', fontsize=15)

ax.set_ylabel('No. of Launches', fontsize=15)

plt.xticks(rotation=45)

plt.show()
fig, ax = plt.subplots(figsize=(20, 10))

ax.set_title('Launch Times by Hour (UTC)', fontsize=20)

sns.countplot(x='Hour', data=df, order=df['Hour'].unique().sort())

ax.set_xlabel('Hour in UTC', fontsize=15)

ax.set_ylabel('No. of Launches', fontsize=15)

plt.xticks(rotation=45)

plt.show()