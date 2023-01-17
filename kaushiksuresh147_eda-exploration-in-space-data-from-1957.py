import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly

from plotly.subplots import make_subplots

import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
space_df=pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')
space_df.info()
space_df.head(10)
# Looks like unnamed columns are not needed here! So removing those two columns before proceeding further!

space_df=space_df.iloc[:,2:len(space_df.columns)]

space_df
import missingno as mno

print(space_df.shape)

mno.matrix(space_df)
missing=pd.DataFrame(space_df.isna().sum().reset_index())

missing.columns=['Variables','Missing']

missing['Percentage']=(missing['Missing']/space_df.shape[0])*100

missing
#Since the values are in string format! We replace the value with the mode value!

print(space_df[' Rocket'].mode())

#The mode value is 450$M

space_df[' Rocket']=space_df[' Rocket'].fillna('450.0')

Stat=pd.DataFrame(space_df.describe().T)

Stat['Percent']=(Stat['freq']/Stat['count'])*100

Stat
space_df['Country'] = space_df['Location'].apply(lambda x:x.split(',')[-1])

country = space_df.groupby('Country').count()['Detail'].sort_values(ascending=False).reset_index()

country.rename(columns={"Detail":"No of Launches"},inplace=True)

country.head(10).style.background_gradient(cmap='Oranges').hide_index()

space_df[space_df['Company Name']=='RVSN USSR']
ussr=pd.DataFrame(space_df[space_df['Company Name']=='RVSN USSR'][['Status Rocket','Status Mission']].value_counts())

ussr.columns=['Count']

ussr['Percentage']=(ussr['Count']/space_df[space_df['Company Name']=='RVSN USSR'].shape[0])*100

ussr
df_active = space_df[space_df['Status Rocket'] == "StatusActive"]

df_active = df_active.groupby('Company Name').count()['Detail'].sort_values(ascending=False).reset_index()

len(df_active)



companies = space_df.groupby(['Company Name'])['Detail'].count().sort_values(ascending=False).reset_index()

len(companies)



top_20 = companies[1:40]

cmp = space_df.groupby(['Company Name','Status Rocket']).count()['Detail'].reset_index()

cmp = cmp[cmp['Company Name'].isin(top_20['Company Name'])]

active = cmp[cmp['Status Rocket']=="StatusActive"].sort_values('Detail')

retired = cmp[cmp['Status Rocket']!="StatusActive"]

fig = go.Figure()

fig.add_bar(y=active['Detail'],x=active['Company Name'],name='Status Active')

fig.add_bar(y=retired['Detail'],x=retired['Company Name'],name='Status Retired')

fig.update_layout(barmode="stack",title="Companies and Mission Status",yaxis_title="No of Missions")

fig.show()
activers=cmp[cmp['Status Rocket']=='StatusActive']['Company Name'].unique()

activers
map_data = [go.Choropleth( 

           locations = country['Country'],

           locationmode = 'country names',

           z = country["No of Launches"], 

           text = country['Country'],

           colorbar = {'title':'No of Launches'},

           colorscale='purples')]



layout = dict(title = 'Countries wise Rocket Launches', 

             geo = dict(showframe = False, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)

print("The more darker the region denotes more number of launches from that respective country!!")


space_df['day'] = space_df['Datum'].apply(lambda x:x.split()[0])

space_df['Month']=space_df['Datum'].apply(lambda x:x.split()[1])

space_df['year'] = space_df['Datum'].apply(lambda x:x.split()[3])

fig, ax = plt.subplots(figsize=(20, 10))

ax.set_title('No. of Launches by Month', fontsize=20)

order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sns.countplot(x='Month', data=space_df, order=order)

ax.set_xlabel('Month', fontsize=15)

ax.set_ylabel('No. of Launches', fontsize=15)

plt.show()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

df_days = space_df.groupby('day').count()['Detail'].reset_index()



df_days['day'] = pd.Categorical(df_days['day'], categories=days, ordered=True)

df_days = df_days.sort_values('day')

plt.figure(figsize=(11,4))

sns.barplot(x='day', y='Detail', data=df_days)

plt.ylabel('No of launches')

b=plt.title(' Day vs No of launches')
date= space_df.groupby('year').count()['Detail'].reset_index()

plt.figure(figsize=(20,6))

b=sns.barplot(x='year', y='Detail', data=date)

plt.ylabel('no of launches')

plt.title(' No of launches per year')

_=b.set_xticklabels(b.get_xticklabels(), rotation=90, horizontalalignment='right')