## importing libraries:

import numpy as np # Linear Algebra

import pandas as pd # To work with data

from plotly.offline import init_notebook_mode, iplot # Offline visualizations

import plotly.graph_objects as go # Visualizations

import plotly.express as px # Visualizations

import matplotlib.pyplot as plt # Just in case

from plotly.subplots import make_subplots # Subplots in plotly

from wordcloud import WordCloud, STOPWORDS # WordCloud
df = pd.read_csv("../input/311-service-requests-pitt/ServreqPitt.csv") # Loading the Data
df.shape
df.isnull().sum() # Too many null values
df.dropna(inplace=True)
df.head() # a look into the dataset.
# Converting the attribute into datetime object will help in many ways.

df.loc[:,'CREATED_ON'] = pd.to_datetime(df['CREATED_ON']) 



df['Month'] = df['CREATED_ON'].dt.month_name()

df['Day'] = df['CREATED_ON'].dt.day

df['Hour'] = df['CREATED_ON'].dt.hour

df['Weekday'] = df['CREATED_ON'].dt.weekday_name
## Let's have a look at the unique value that each attribute has.

for i in df.columns :

    print(i,':', len(df[i].unique()))
temp = df['REQUEST_ORIGIN'].value_counts().reset_index()

temp.columns=['Request_Origin', 'Count']



fig = make_subplots(

    rows=1, cols=2,

    specs=[[{"type": "xy"},{"type": "domain"}]]

)



fig.add_trace(go.Bar(x=temp['Request_Origin'], y=temp['Count']), 1,1)

fig.add_trace(go.Pie(labels=temp['Request_Origin'], values=temp['Count'], pull=[0.1,0,0,0,0,0,0,0]), 1,2)

fig.update_traces(marker=dict(line=dict(color='#000000', width=2)))

fig.update_layout(title='Distribution of requests amongst Request Origins:')

iplot(fig)
temp = df['DEPARTMENT'].value_counts().reset_index()

temp.columns=['Department', 'Count']



fig = px.bar(temp, 'Department', 'Count')

fig.update_layout(title='Number of requests from departments:', xaxis_tickangle=-25)

iplot(fig)
temp = df['COUNCIL_DISTRICT'].value_counts().reset_index().head(20)

temp.columns=['Council_District', 'Count']



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'domain'}]])

fig.add_trace(go.Bar(x=temp['Council_District'], y=temp['Count']), 1,1)

fig.add_trace(go.Pie(labels=temp['Council_District'], values=temp['Count']), 1,2)

fig.update_traces(textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))

fig.update_layout(title='Distribution of requests amongst council districts:')

iplot(fig)
temp = df['WARD'].value_counts().reset_index()

temp.columns=['Ward', 'Count']



fig = px.bar(temp, 'Ward', 'Count', color='Count')

fig.update_layout(title='Request distributions amongst Wards:', xaxis_tickangle=-25)

iplot(fig)
temp = df['PUBLIC_WORKS_DIVISION'].value_counts().reset_index().head(20)

temp.columns=['Public Work Division', 'Count']



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'domain'}]])

fig.add_trace(go.Bar(x=temp['Public Work Division'], y=temp['Count']), 1,1)

fig.add_trace(go.Pie(labels=temp['Public Work Division'], values=temp['Count']), 1,2)

fig.update_traces(textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))

fig.update_layout(title='Distribution of requests amongst Public Work Divisions:')

iplot(fig)
temp = df['POLICE_ZONE'].value_counts().reset_index().head(20)

temp.columns=['Police Zone', 'Count']



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'xy'}, {'type':'domain'}]])

fig.add_trace(go.Bar(x=temp['Police Zone'], y=temp['Count']), 1,1)

fig.add_trace(go.Pie(labels=temp['Police Zone'], values=temp['Count'], textinfo='label+percent'), 1,2)

fig.update_traces(textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))

fig.update_layout(title='Distribution of requests amongst Police Zones:')

iplot(fig)
temp = df['PUBLIC_WORKS_DIVISION'].value_counts().reset_index()

temp.columns=['Public Work Division', 'Count']

fig = go.Figure(data=[

    go.Bar(x=temp['Count'], y=temp['Public Work Division'],

           marker=dict(color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1)),

          orientation='h'

          )

])

fig.update_layout(title='Distrbution of request amongst Public Work Divisions')

iplot(fig)
temp = df['PLI_DIVISION'].value_counts().reset_index()

temp.columns=['PLI_DIVISION', 'Count']

fig = go.Figure(data=[

    go.Bar(x=temp['Count'], y=temp['PLI_DIVISION'],

           marker=dict(color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1)),

          orientation='h'

          )

])

fig.update_layout(title='Distrbution of request amongst PLI divisions')

iplot(fig)
temp = df['POLICE_ZONE'].value_counts().reset_index()

temp.columns=['Police Zone', 'Count']

fig = go.Figure(data=[

    go.Bar(x=temp['Count'], y=temp['Police Zone'],

           marker=dict(color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1)),

          orientation='h'

          )

])

fig.update_layout(title='Distrbution of request amongst Police Zone')

iplot(fig)
temp = df['FIRE_ZONE'].value_counts().reset_index().head(20)

temp

temp.columns=['Fire Zone', 'Count']

fig = go.Figure(data=[

    go.Bar(x=temp['Count'], y=temp['Fire Zone'],

           marker=dict(color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1)),

          orientation='h'

          )

])

fig.update_layout(title='Fire Zones that recieve most requests.')

iplot(fig)
temp = df['REQUEST_TYPE'].value_counts().reset_index()

temp.columns=['Request_Type', 'Count']



fig = px.bar(temp.head(20), 'Request_Type', 'Count')

fig.update_layout(title='Most reported Service Requests:')

iplot(fig)

temp = df['STATUS'].value_counts().reset_index()

temp.columns=['Status', 'Count']

fig = go.Figure(data=[

    go.Pie(labels=temp['Status'], values=temp['Count'])

])

fig.update_traces(textfont_size=20, marker=dict(line=dict(color='#000000', width=2)))

fig.update_layout(title='Distribution of request status:')

iplot(fig)
temp = df['REQUEST_TYPE'].value_counts().reset_index()

temp.columns=['Request_Type', 'Count']

top_requests = temp.head(20)['Request_Type'].tolist()

temp=df[df['REQUEST_TYPE'].isin(top_requests)]



temp=temp.groupby(by=['REQUEST_TYPE', 'STATUS'])['REQUEST_ID'].count()

temp = temp.unstack().fillna(0).reset_index()

temp.columns=['Request Type','0','1','3']

temp = pd.melt(temp, id_vars='Request Type', value_vars=['0','1','3'])

fig = px.bar(temp, 'Request Type', 'value', color='variable')

fig.update_layout(title='Request Status of top reported Request Types')

iplot(fig)
temp = df.set_index('CREATED_ON').sort_index()

temp = temp.resample('W')['REQUEST_ID'].count().reset_index()

fig = px.line(temp, 'CREATED_ON', 'REQUEST_ID')

fig.update_layout(title = 'Trend of 311 calls. (By Week)')

iplot(fig)
temp = df['Month'].value_counts().reset_index()

temp.columns=['Month', 'Count']

temp.sort_values(by='Count', inplace=True)

fig = px.scatter(temp, 'Month', 'Count', size='Count', color='Count')

fig.update_layout(title='Requests by months:')

iplot(fig)
temp = df['Day'].value_counts().reset_index()

fig = px.scatter(temp, 'index', 'Day', color='Day', size='Day',

          labels={'index':'Day', 'Day':'Requests'})

fig.update_layout(title='Requests by day of month')

iplot(fig)
temp = df['Weekday'].value_counts().reset_index()

fig = px.scatter(temp, 'index', 'Weekday', color='Weekday', size='Weekday',

          labels={'index':'Weekday', 'Weekday':'Requests'})

fig.update_layout(title='Requests by Weekday')
temp = df['Hour'].value_counts().reset_index()

fig = px.scatter(temp, 'index', 'Hour', color='Hour', size='Hour',

          labels={'index':'Hour', 'Hour':'Requests'})

fig.update_layout(title='Requests by hour of day')

iplot(fig)
temp = df['REQUEST_TYPE'].value_counts().reset_index()

top_types = temp.head(20)['index'].tolist()

del temp

df1 = df[df['REQUEST_TYPE'].isin(top_types)]

df1 = df1.groupby(by=['Month', 'REQUEST_TYPE'])['REQUEST_ID'].count().unstack().reset_index()

vars_list = list(df1.columns)[1:]

df1 = pd.melt(df1, id_vars='Month', value_vars=vars_list)

df1.columns=['Month','Request Type', 'Requests']



fig = px.scatter(df1, x='Request Type', y='Requests', color='Month')

fig.update_layout(title='Distribution of Requests of top 20 most reported Tequest Types :')

iplot(fig)
df1 = df.set_index('CREATED_ON')

df1 = df1[['REQUEST_ORIGIN']]

df1 = pd.get_dummies(df1)

df1 = df1.resample('M').sum()

df1 = df1.cumsum()

df1.reset_index(inplace=True)

df1 = pd.melt(df1, id_vars=['CREATED_ON'], value_vars=['REQUEST_ORIGIN_Call Center',

 'REQUEST_ORIGIN_Control Panel',

 'REQUEST_ORIGIN_Email',

 'REQUEST_ORIGIN_QAlert Mobile iOS',

 'REQUEST_ORIGIN_Report2Gov Android',

 'REQUEST_ORIGIN_Report2Gov Website',

 'REQUEST_ORIGIN_Report2Gov iOS',

 'REQUEST_ORIGIN_Text Message',

 'REQUEST_ORIGIN_Twitter',

 'REQUEST_ORIGIN_Website'])

df1.columns = ['Created_On', 'Request_Origin', 'Requests']

df1.loc[:,'Request_Origin'] = df1['Request_Origin'].apply(lambda x : str(x).split('REQUEST_ORIGIN_')[1])

df1.loc[:,'Created_On'] = df1.loc[:,'Created_On'].dt.strftime('%Y-%m-%d')



fig = px.bar(df1, 'Request_Origin', 'Requests', animation_frame='Created_On')

fig.update_layout(title='Requests distribution amongst Request Origins throughout time:')

iplot(fig)
temp = df['REQUEST_TYPE'].value_counts().reset_index()

top_types = temp.head(20)['index'].tolist()

del temp

df1 = df.set_index('CREATED_ON')

df1 = df1[df1['REQUEST_TYPE'].isin(top_types)]['REQUEST_TYPE']

df1 = pd.get_dummies(df1)

df1 = df1.resample('M').sum()

df1 = df1.cumsum()

df1.reset_index(inplace=True)

df1 = pd.melt(df1, id_vars='CREATED_ON', value_vars=top_types)

df1.columns = ['Created_On', 'Request_Type', 'Requests']

df1.loc[:,'Created_On'] = df1.loc[:,'Created_On'].dt.strftime('%Y-%m-%d')



fig = px.bar(df1, 'Request_Type', 'Requests', animation_frame='Created_On')

fig.update_layout(title='Requests distribution amongst most reported Request Types throughout time:')

iplot(fig)