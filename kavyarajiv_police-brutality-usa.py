from IPython.display import Image

import os

!ls ../input/

Image("../input/cops-brutality/img2.jpg")
import pandas as pd

import numpy as np

import plotly.figure_factory as ff

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

d_f=pd.read_csv('../input/police-brutality/fatal-police-shootings-data.csv')

d_f
d_f.isnull().sum()
d_f.armed.replace({np.nan:'Unarmed'},inplace=True)

d_f.age.replace({np.nan:0},inplace=True)

d_f.race.replace({np.nan:'Other'},inplace=True)

d_f.flee.replace({np.nan:'Undetermined'},inplace=True)

d_f.dropna(axis=0,subset=['gender'],inplace=True)
d_f.isnull().sum()
d_f
df=d_f.age[d_f.age.drop(labels=0)]

fig=px.histogram(df,nbins=20,color_discrete_sequence=['Red'],labels={'value':'Age'},template='plotly_dark',title= ' Age groups affected by police brutality ')

fig.show()
d_f['year']=pd.to_datetime(d_f['date']).dt.year

d_year=d_f['year'].value_counts().reset_index().rename(columns={'index':'year','year':'count'}).sort_values(by='year')
fig=go.Figure(go.Scatter(x=d_year['year'],y=d_year['count'],mode='markers',marker_color=['red','yellow','green','blue','pink','orange'],marker_size=[20,20,20,20,20,20]))

fig.update_layout(title='Yearly Deaths',template="plotly_white")

fig.show()
d_f['month']=pd.to_datetime(d_f['date']).dt.month

d_month=d_f['month'].value_counts().reset_index().rename(columns={'index':'month','month':'count'}).sort_values(by='month')

fig=px.bar(x=d_month['count'],y=d_month['month'],color_discrete_sequence=['mediumvioletred'],labels={'x':'Month','y':'Count'},title='Monthly Deaths',orientation='h',template='seaborn')

fig.show()
d_1=d_f.manner_of_death.value_counts()

d_death=d_1.reset_index().rename(columns={'index':'Manner of Death','manner_of_death':'Count'})

fig=px.pie(data_frame=d_death,names=d_death['Manner of Death'],values=d_death['Count'].values, color_discrete_sequence=px.colors.sequential.RdBu,title='Manner of death')

fig.show()
d_3=d_f.groupby(['year','gender'])[['gender']].count()

d_gender=d_3.rename(columns={'gender':'count'}).reset_index()

px.bar(data_frame=d_gender,x=d_gender['year'],y=d_gender['count'],color=d_gender['gender'],barmode='group',title='Yearly deaths as per gender',template='ggplot2')
d_race=d_f.groupby(['year','race'])[['race']].count().rename(columns={'race':'count'}).reset_index(level=1).replace(['W','B','A','N','H','O'] ,['White', 'Black',' Asian',' Native American','Hispanic','Other'])
d1=d_race[d_race.index==2015]

d2=d_race[d_race.index==2016]

d3=d_race[d_race.index==2017]

d4=d_race[d_race.index==2018]

d5=d_race[d_race.index==2019]

d6=d_race[d_race.index==2020]

fig=make_subplots(rows=6, cols=1, shared_xaxes=True,vertical_spacing=0.05)

fig.add_trace(go.Scatter(x=d1['race'],y=d1['count'],name='2015'),row=1,col=1)

fig.add_trace(go.Scatter(x=d2['race'],y=d2['count'],name='2016'),row=2,col=1)

fig.add_trace(go.Scatter(x=d3['race'],y=d3['count'],name='2017'),row=3,col=1)

fig.add_trace(go.Scatter(x=d4['race'],y=d4['count'],name='2018'),row=4,col=1)

fig.add_trace(go.Scatter(x=d5['race'],y=d5['count'],name='2019'),row=5,col=1)

fig.add_trace(go.Scatter(x=d6['race'],y=d6['count'],name='2020'),row=6,col=1)

fig.update_layout(height=1000,width=700)
d_sign=d_f.signs_of_mental_illness.value_counts().reset_index().rename(columns={'index':'signs_of_mental_illness','signs_of_mental_illness':'count'})

d_threat=d_f.threat_level.value_counts().reset_index().rename(columns={'index':'threat_level','threat_level':'count'})
fig=make_subplots(rows=1,cols=2,specs=[[{'type':'domain'},{'type':'domain'}]])

fig.add_trace(go.Pie(labels=d_sign['signs_of_mental_illness'],values=d_sign['count'],name='mental illness'),row=1,col=1)

fig.add_trace(go.Pie(labels=d_threat['threat_level'],values=d_threat['count'],name='Threat level'),row=1,col=2)

fig.update_traces(hole=0.3)
d_state=d_f.state.value_counts().reset_index().rename(columns={'index':'state','state':'count'})
fig =go.Figure(data=go.Choropleth(locations=d_state['state'],z =d_state['count'],locationmode = 'USA-states',colorscale = 'blues',colorbar_title = "No of deaths"))

fig.update_layout(title_text = 'Deaths in each state',geo_scope='usa')