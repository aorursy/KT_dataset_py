

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as offline

import plotly.graph_objs as go



offline.init_notebook_mode(connected=True)



#OverView of the data file

apps_data = pd.read_csv("../input/googleplaystore.csv")

apps_data.head(10)
#Cleaning the file for better visualization results

apps_data=apps_data.dropna()
app_categories=apps_data['Category'].unique() #list of all categories

#print(app_categories)

app_category_dict=apps_data['Category'].value_counts() #Each category and there count(series)

print(app_category_dict)
values = app_category_dict.keys().tolist()



counts = app_category_dict.tolist()



x = counts

y = values

trace_bar =[go.Bar(x = counts,y = values,orientation='h',marker=dict(color=counts,colorscale='Viridis'))]



layout_bar = go.Layout(title = 'Count of Apps in each Category found in Play Store',

                   titlefont = dict(family = 'Arial',size = 20,color = '#8B0000' ),

                   height=900,

                   yaxis=dict(title='Category',titlefont=dict(family='Arial, sans-serif',size=18,color='grey'),showticklabels=True,

                    tickfont=dict(family='Old Standard TT, serif',size=10,color='#4E1C11')),

                   xaxis=dict(title='Count(In Numbers)',titlefont=dict(family='Arial, sans-serif',size=18,color='grey'),

                              domain=[0.1,0.6]))



fig = go.Figure(data=trace_bar,layout=layout_bar)

offline.iplot(fig)
free_app=apps_data.loc[apps_data['Type'] == 'Free']

app_category_dict_free=free_app['Category'].value_counts() #Each category and there count(series)

#print(app_category_dict_free)



paid_app=apps_data.loc[apps_data['Type'] == 'Paid']

app_category_dict_paid=paid_app['Category'].value_counts() #Each category and there count(series)

#print(app_category_dict_paid)
trace1 =go.Bar(x = app_category_dict_free.tolist(), y = app_category_dict_free.keys().tolist(),

                orientation='h',name='Free',marker={'color': '#0B405F'})



trace2 =go.Bar(x = app_category_dict_paid.tolist(), y = app_category_dict_paid.keys().tolist(),

                orientation='h',xaxis='x2',yaxis='y2',name='Paid',marker={'color': '#06848A'})



layout1 = go.Layout(title = 'Count of Apps in Different categories in Play Store ( Free Vs. Paid )',

                   titlefont = dict(family = 'serif',size =25,color = '#8B0000' ),

                   height=900,

                   

                   yaxis=dict(title='Category',titlefont=dict(family='Arial, sans-serif',size=18,color='Grey'),

                                showticklabels=True,

                                tickwidth=1,

                                tickfont=dict(family='Calibri',size=12,color='#4E1C11')),

                   

                   xaxis=dict(title='Count (In Numbers) [Free Apps]',tickwidth=1,domain=[0.1,0.6],titlefont=dict(family='Arial, sans-serif',size=14,color='#0B405F')),

                   xaxis2=dict(title='Count (In Numbers) [Paid Apps]',domain=[0.7,1.0],titlefont=dict(family='Arial, sans-serif',size=14,color='#06848A')),

                   yaxis2=dict(anchor='x2',title='Category',titlefont=dict(family='Arial, sans-serif',size=18,color='Grey'),

                                showticklabels=True,

                                tickwidth=1,

                                tickfont=dict(family='Calibri',size=12,color='#4E1C11'))

                  )



fig = go.Figure(data=[trace1,trace2],layout=layout1)

offline.iplot(fig)
genre_series = apps_data['Genres']

genre_list=[]

for values in genre_series:

    s=values.split(';')

    genre_list.extend(s)

#print(genre_list)

genre_data=pd.Series(genre_list).value_counts()



trace_pie=[go.Pie(labels = genre_data.keys().tolist(),values = genre_data.tolist(),

                 hoverinfo = 'label+value+percent', 

               

               textinfo = 'percent', 

               textfont = dict(size = 10),

               

               marker = dict(

                             line = dict(color = 'black', 

                                         width = 1)

                            ),hole = 0.08

             )]

layout_pie = go.Layout(title = 'Percentage of Apps in different Genre',

                   titlefont = dict(family = 'serif',size = 25,color = 'Black' ),

                   height=900)



fig = go.Figure(data=trace_pie,layout=layout_pie)

offline.iplot(fig)
apps_data['Installs'] = apps_data['Installs'].astype(str)

apps_data['Installs'] = apps_data['Installs'].str.replace(r'\D', '').astype(int)

dataframe1=apps_data.groupby('Content Rating')[['Installs']].sum()

#print(dataframe1)

trace_scatter=[go.Scatter(x=dataframe1.index.tolist(),

                 y=dataframe1.Installs.tolist() ,

                 hoverinfo = "x+y",

                 mode='markers',

                 marker=dict(size=14))]



layout_scatter = go.Layout(title = 'Installs in each Content Rating',

                   titlefont = dict(family = 'serif',size = 25,color = 'Black' ),

                   height=900,plot_bgcolor = '#E2E2E2' )



fig_scatter = go.Figure(data=trace_scatter,layout=layout_scatter)

offline.iplot(fig_scatter)
category_list=apps_data['Category'].unique()

N=len(category_list)

#print(N)

data_box=[{'name':category_list[i],

       'y':apps_data.loc[apps_data['Category'] == category_list[i] ,'Rating'],

       'type' : 'box'

    }for i in range(int(N))]



layout_box=go.Layout(title='Rating Analysis for Each Category',titlefont=dict(family = 'serif',size = 25,color = 'Black' ),

                   height=900,

                    yaxis=dict(title='Rating',titlefont=dict(family='Arial, sans-serif',size=18,color='grey'),showticklabels=True,

                    tickfont=dict(family='serif',size=15,color='#4E1C11')),

                   xaxis=dict(showticklabels=True,tickangle=30,tickwidth=0.2,

                              tickfont=dict(family='serif',size=10,color='#4E1C11')))

fig_box=go.Figure(data=data_box,layout=layout_box)

offline.iplot(fig_box)
apps_data['Price']=apps_data['Price'].astype(str)

apps_data['Price'] = apps_data['Price'].map(lambda x: x.strip('$'))

apps_data['Price']=apps_data['Price'].astype(float)

dataframe2 = apps_data[apps_data['Type']=='Paid'].groupby('Category')[['Price']].max()

dataframe3 = apps_data[apps_data['Type']=='Paid'].groupby('Category')[['Price']].min()

trace_max=go.Scatter(x=dataframe2.index.tolist(),

                    y=dataframe2.Price.tolist(),

                    name='Highest priced App',

                    line=dict(color = '#490F0F',

                width = 4))



trace_min=go.Scatter(x=dataframe3.index.tolist(),

                    y=dataframe3.Price.tolist(),

                    name='Least priced App',

                    line=dict(color = '#F46523',

                width = 4))

data_lines=[trace_max,trace_min]

layout_lines=go.Layout(title='Highest & Least Priced App in each Category',titlefont=dict(family = 'serif',size = 25,color = 'Black' ),

                   height=900,

                    yaxis=dict(title='Price($)',titlefont=dict(family='Arial, sans-serif',size=18,color='grey'),showticklabels=True,

                    tickfont=dict(family='serif',size=15,color='#4E1C11')),

                   xaxis=dict(showticklabels=True,tickangle=30,tickwidth=0.2,

                              tickfont=dict(family='serif',size=10,color='#4E1C11')))

fig_lines=go.Figure(data=data_lines,layout=layout_lines)

offline.iplot(fig_lines)