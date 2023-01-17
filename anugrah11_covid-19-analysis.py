# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px 

import plotly.graph_objects as go

import warnings 

warnings.filterwarnings('ignore')

import folium
df1 = pd.read_excel("/kaggle/input/covid19-updated/Corono Virus.xlsx", sheet_name='Confirmed')

df2 = pd.read_excel("/kaggle/input/covid19-updated/Corono Virus.xlsx", sheet_name='Recovered')

df3 = pd.read_excel("/kaggle/input/covid19-updated/Corono Virus.xlsx", sheet_name='Deaths')
confirm = df1['3/18/20'].sum()

recovered = df2['3/18/20'].sum()

deaths = df3['3/18/20'].sum()

active = confirm - recovered - deaths



list_all = [recovered,deaths,active]



colors = ['gold', 'red', 'lightgreen'] 

fig = go.Figure(data=[go.Pie(labels=['Recovered', 'Deaths', 'Active'],

                             values=list_all)])

fig.update_traces(hoverinfo='label+value', textinfo='label+percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))



fig.show()
new_df1_time = df1.transpose().reset_index().drop([0,1,2,3,4,57],axis=0)

new_df1_feat = df1.transpose().reset_index().loc[[1,2,3,4]]

new_df1_time['index']=pd.to_datetime(new_df1_time['index'])

list_province = new_df1_feat.loc[1][1:].values
fig = go.Figure()



fig.update_layout(

    width=900,

    height=600,

    autosize=False,

    margin=dict(t=50, b=0, l=0, r=10),

)



for item in range(475):

    fig.add_trace(go.Scatter(x=new_df1_time['index'],

                         y=new_df1_time[item],

                             mode='lines+markers',

                             marker=dict(colorscale='Viridis',line_width=0.1),

                             line=dict(width=3,smoothing=True),

                             name=new_df1_feat[item][1]))

    

# Make buttons for drawdowns

list_dict = []

list_dict.append(dict(label="All",

         method="update",

         args=[{"visible": [True]*(len(list_province))},

               {"title": "Total Covid-19 Confirm all Province/Country",

                "annotations": []}]))



for i,item in enumerate(list_province):

    visible_list = [False]*len(list_province)

    visible_list[i]=True

    list_dict.append(dict(label=item,

         method="update",

         args=[{"visible": visible_list},

               {"title": 'Total Covid-19 Confirmed',

                "annotations": []}]))

    

# Add drawdowns

fig.update_layout(title ={'text':'Trend Covid-19 Confirm for each Province/Country',

                          'yanchor':'top'},

                  xaxis_title = 'Date',

                  yaxis_title = 'Total Confirm',

                  font=dict(family="Arial,Roboto"),

                  showlegend=True,

                  legend_title='<b> Country/Province : </b>',

                  legend_orientation="h",

                  updatemenus=[dict(active=0,

                                    buttons= list_dict

                )])



fig.show()
new_df2_time = df2.transpose().reset_index().drop([0,1,2,3,4,57],axis=0)

new_df2_feat = df2.transpose().reset_index().loc[[1,2,3,4]]

new_df2_time['index']=pd.to_datetime(new_df2_time['index'])
fig = go.Figure()



fig.update_layout(

    width=900,

    height=600,

    autosize=False,

    margin=dict(t=50, b=0, l=0, r=10),

)



for item in range(475):

    fig.add_trace(go.Scatter(x=new_df2_time['index'],

                         y=new_df2_time[item],

                             mode='lines+markers',

                             marker=dict(colorscale='Viridis',line_width=0.1),

                             line=dict(width=3,smoothing=True),

                             name=new_df2_feat[item][1]))

    

# Make buttons for drawdowns

list_dict = []

list_dict.append(dict(label="All",

         method="update",

         args=[{"visible": [True]*(len(list_province))},

               {"title": "Total Covid-19 Recovered All Province/Country",

                "annotations": []}]))



for i,item in enumerate(list_province):

    visible_list = [False]*len(list_province)

    visible_list[i]=True

    list_dict.append(dict(label=item,

         method="update",

         args=[{"visible": visible_list},

               {"title": 'Total Covid-19 Recovered',

                "annotations": []}]))

    

# Add drawdowns

fig.update_layout(title ={'text':'Trend Covid-19 Recovered All Province/Country',

                          'yanchor':'top'},

                  xaxis_title = 'Date',

                  yaxis_title = 'Total Recovered',

                  font=dict(family="Arial,Roboto"),

                  showlegend=True,

                  legend_title='<b> Country/Province : </b>',

                  legend_orientation="h",

                  updatemenus=[dict(active=0,

                                    buttons= list_dict

                )])



fig.show()
new_df3_time = df3.transpose().reset_index().drop([0,1,2,3,4,57],axis=0)

new_df3_feat = df3.transpose().reset_index().loc[[1,2,3,4]]

new_df3_time['index']=pd.to_datetime(new_df3_time['index'])
fig = go.Figure()



fig.update_layout(

    width=900,

    height=600,

    autosize=False,

    margin=dict(t=50, b=0, l=0, r=10),

)



for item in range(475):

    fig.add_trace(go.Scatter(x=new_df3_time['index'],

                         y=new_df3_time[item],

                             mode='lines+markers',

                             marker=dict(colorscale='Viridis',line_width=0.1),

                             line=dict(width=3,smoothing=True),

                             name=new_df3_feat[item][1]))

    

# Make buttons for drawdowns

list_dict = []

list_dict.append(dict(label="All",

         method="update",

         args=[{"visible": [True]*(len(list_province))},

               {"title": "Total Covid-19 Deaths All Province/Country",

                "annotations": []}]))



for i,item in enumerate(list_province):

    visible_list = [False]*len(list_province)

    visible_list[i]=True

    list_dict.append(dict(label=item,

         method="update",

         args=[{"visible": visible_list},

               {"title": 'Total Covid-19 Deaths',

                "annotations": []}]))

    

# Add drawdowns

fig.update_layout(title ={'text':'Trend Covid-19 Deaths All Province/Country',

                          'yanchor':'top'},

                  xaxis_title = 'Date',

                  yaxis_title = 'Total Deaths',

                  font=dict(family="Arial,Roboto"),

                  showlegend=True,

                  legend_title='<b> Country/Province : </b>',

                  legend_orientation="h",

                  updatemenus=[dict(active=0,

                                    buttons= list_dict

                )])



fig.show()
df_geo_confirm = df1[['Province/State', 'Lat', 'Long','3/18/20']]

df_geo_confirm.rename(columns={'3/18/20' : 'Total Confirm'},inplace=True)



df_geo_recovered = df2[['Province/State', 'Lat', 'Long','3/18/20']]

df_geo_recovered.rename(columns={'3/18/20' : 'Total Recovered'},inplace=True)



df_geo_deaths = df3[['Province/State', 'Lat', 'Long','3/18/20']]

df_geo_deaths.rename(columns={'3/18/20' : 'Total Deaths'},inplace=True)
fig = px.scatter_geo(df_geo_confirm,lon='Long', lat='Lat',color='Total Confirm', hover_name='Province/State', 

                     size='Total Confirm', projection= 'natural earth')

fig.show()
fig = px.scatter_geo(df_geo_recovered,lon='Long', lat='Lat',color='Total Recovered', hover_name='Province/State', 

                     size='Total Recovered', projection= 'natural earth')

fig.show()
fig = px.scatter_geo(df_geo_deaths,lon='Long', lat='Lat',color='Total Deaths', hover_name='Province/State', 

                     size='Total Deaths', projection= 'natural earth')

fig.show()
a= df1[df1['Province/State'] == 'Indonesia']

b= df2[df2['Province/State'] == 'Indonesia']

c= df3[df3['Province/State'] == 'Indonesia']





d= df1[df1['Province/State'] == 'Hubei, China']

e= df2[df2['Province/State'] == 'Hubei, China']

f= df3[df3['Province/State'] == 'Hubei, China']



df_indo = pd.concat([a,b,c])

df_hubei = pd.concat([d,e,f])
fig = go.Figure(go.Scatter(x=new_df1_time['index'],

                         y=new_df1_time[154],

                             mode='lines+markers',

                             marker=dict(colorscale='Viridis',line_width=0.1),

                             line=dict(width=3,smoothing=True),

                             name='Hubei, China'))



fig.show()
a = df_hubei['3/18/20'].values

list_all = [a[1],a[2],(a[0]-a[1]-a[2])]



colors = ['darkcyan', 'gold', 'limegreen'] 

fig = go.Figure(data=[go.Pie(labels=['Recovered', 'Deaths', 'Active'],

                             values=list_all)])

fig.update_traces(hoverinfo='label+value', textinfo='label+percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig = go.Figure(go.Scatter(x=new_df1_time['index'],

                         y=new_df1_time[58],

                             mode='lines+markers',

                             marker=dict(colorscale='Viridis',line_width=0.1),

                             line=dict(width=3,smoothing=True),

                             name='Hubei, China'))



fig.show()
a = df_indo['3/18/20'].values

list_all = [a[1],a[2],(a[0]-a[1]-a[2])]



colors = ['darkcyan', 'gold', 'limegreen'] 

fig = go.Figure(data=[go.Pie(labels=['Recovered', 'Deaths', 'Active'],

                             values=list_all)])

fig.update_traces(hoverinfo='label+value', textinfo='label+percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
from IPython.display import Image

Image("/kaggle/input/imagecovid19/COVID-19 Analogy.jpg")