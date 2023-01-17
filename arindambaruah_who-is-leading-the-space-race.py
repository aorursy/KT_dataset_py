import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

plotly.offline.init_notebook_mode (connected = True)
df=pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv',index_col=0)

df.drop('Unnamed: 0.1',axis=1,inplace=True)

df.head()
df.isna().any()
df[' Rocket'].isna().value_counts()
df.dtypes
df['Launch date']=pd.to_datetime(df['Datum'])

df['Launch date']=df['Launch date'].astype(str)
df['Launch date']=df['Launch date'].str.split(' ',expand=True)[0]

df['Launch date']=pd.to_datetime(df['Launch date'])
df[' Rocket']=df[' Rocket'].str.replace(',','')

df[' Rocket']=df[' Rocket'].astype(float)
df['Status Rocket']=df['Status Rocket'].str.replace('Status','')
df.drop('Datum',axis=1,inplace=True)

df.head()
df['Count']=1

df_status=df.groupby('Status Mission')['Count'].sum().reset_index()



fig=px.pie(df_status,values='Count',names='Status Mission',hole=0.4)



fig.update_layout(title='Mission status',title_x=0.5,

                  annotations=[dict(text='Status',font_size=15, showarrow=False,height=800,width=900)])

fig.update_traces(textfont_size=15,textinfo='percent')

fig.show()
sns.catplot('Status Rocket',kind='count',data=df,aspect=2,height=5,palette='viridis')

plt.title('Current status of rockets',size=25)

plt.xlabel('Rocket status',size=15)

df_comps=df.groupby('Company Name')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

df_comps=df_comps.head(20)

sns.catplot('Company Name','Count',data=df_comps,palette='coolwarm',kind='bar',aspect=2,height=8)

plt.title('Top 20 space companies',size=25)

plt.xticks(rotation=80,size=15)

plt.xlabel('Company name',size=20)

plt.yticks(size=15)
df['Country']=df['Location'].str.split(', ').str[-1]
df_countries=df.groupby('Country')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)

df_countries=df_countries.head(10)
sns.catplot('Country','Count',data=df_countries,aspect=2,height=8,kind='bar',palette='winter')



plt.title('Top 10 nations leading the space launches',size=25)

plt.xticks(size=15,rotation=45)

plt.xlabel('Country',size=15)

plt.ylabel('Count',size=15)

plt.yticks(size=15)
df_cost=df.dropna()

plt.figure(figsize=(10,8))

sns.distplot(df_cost[' Rocket'],color='green')

plt.title('Mission cost in million $',size=25)

plt.xlabel('Cost of mission (million $)',size=15)
df_cost_comp=df_cost.groupby('Company Name')[' Rocket'].sum().reset_index().sort_values(by=' Rocket',ascending=False)



df_cost_comp[' Rocket']=df_cost_comp[' Rocket']/1000 #Converting costs to billion $


sns.catplot('Company Name',' Rocket',data=df_cost_comp,aspect=2,height=8,kind='point')

plt.xticks(size=15,rotation=45)

plt.xlabel('Company name',size=20)

plt.ylabel('Money spent (in billion $)',size=20)

plt.yticks(size=15)

plt.title('Money spent on space missions',size=25)
df['Year']=df['Launch date'].dt.year
df_year=df.groupby('Year')['Count'].sum().reset_index()



fig=px.line(df_year,y='Count',x='Year',height=800,width=1000)

fig.update_layout(title='Number of missions each year',font_size=20,title_x=0.5)

fig.show()
df_latest=df[df['Year']==2020]
sns.catplot('Company Name',data=df_latest,kind='count',aspect=2,height=8)

plt.yticks(np.arange(20))

plt.title('2020 launches',size=25)

plt.xlabel('Company name',size=20)

plt.xticks(size=15,rotation=45)

plt.yticks(size=15)

plt.ylabel('Number of missions',size=15)
df_nasa=df[df['Company Name']=='NASA']

df_isro=df[df['Company Name']=='ISRO']

df_esa=df[df['Company Name']=='ESA']

df_rosc=df[df['Company Name']=='Roscosmos']
fig1=plt.figure(figsize=(10,15))

ax1=fig1.add_subplot(221)

sns.countplot('Status Mission',data=df_nasa,ax=ax1)

ax1.set_title('NASA success rate = {0:.2f}%'.format(100*df_nasa['Status Mission'].value_counts()[0]/df_nasa.shape[0]),size=15)



ax2=fig1.add_subplot(222)

sns.countplot('Status Mission',data=df_rosc,ax=ax2,palette='summer')

ax2.set_title('Roscosmos success rate = {0:.2f}%'.format(100*df_rosc['Status Mission'].value_counts()[0]/df_rosc.shape[0]),size=15)





ax3=fig1.add_subplot(223)

sns.countplot('Status Mission',data=df_esa,ax=ax3,palette='winter')

ax3.set_title('ESA success rate = {0:.2f}%'.format(100*df_esa['Status Mission'].value_counts()[0]/df_esa.shape[0]),size=15)





ax4=fig1.add_subplot(224)

sns.countplot('Status Mission',data=df_isro,ax=ax4,palette='coolwarm')

ax4.set_title('ISRO success rate = {0:.2f}%'.format(100*df_rosc['Status Mission'].value_counts()[0]/df_isro.shape[0]),size=15)



map_data = [go.Choropleth( 

           locations = df_countries['Country'],

           locationmode = 'country names',

           z = df_countries["Count"], 

           text = df_countries['Country'],

           colorbar = {'title':'No. of Launches'},

           colorscale='temps')]



layout = dict(title = 'Missions per country', title_x=0.5,

             geo = dict(showframe = False, 

                       projection = dict(type = 'equirectangular')))



world_map = go.Figure(data=map_data, layout=layout)

iplot(world_map)