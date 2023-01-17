



#Importing relevant libraries

import numpy as np 

import pandas as pd 

import plotly as py

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



##Importing data into notebook

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#Reading the data by pandas..Trying this you may have to change location according to local location



corona_data=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

#Viewing first three rows of data for quick insight

corona_data.head(3)

#Viewing last two rows of data

corona_data.tail(2)

choro_map=px.choropleth(corona_data, 

                    locations="Country/Region", 

                    locationmode = "country names",

                    color="Confirmed", 

                    hover_name="Country/Region", 

                    animation_frame="ObservationDate"

                   )



choro_map.update_layout(

    title_text = 'Global Spread of Coronavirus',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

choro_map.show()
pie_chart = px.pie(corona_data, values = 'Confirmed',names='Country/Region', height=600)

pie_chart.update_traces(textposition='inside', textinfo='percent+label')



pie_chart.update_layout(

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))



pie_chart.show()
#Manipulating the dataframe

top10 = corona_data.groupby(['Country/Region', 'ObservationDate']).sum().reset_index().sort_values('Confirmed', ascending=False)

top10  = top10.drop_duplicates(subset = ['Country/Region'])

top10 = top10.iloc[0:10]

pie_chart_top10 = px.pie(top10, values = 'Confirmed',names='Country/Region', height=600)

pie_chart_top10.update_traces(textposition='inside', textinfo='percent+label')



pie_chart_top10.update_layout(

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))



pie_chart_top10.show()
#Manipulating the dataframe

last20 = corona_data.groupby(['Country/Region', 'ObservationDate']).sum().reset_index().sort_values('Confirmed', ascending=False)

last20  = last20.drop_duplicates(subset = ['Country/Region'])

last20 = last20.iloc[-20:-1]

last20
pie_chart_last20 = px.pie(last20, values = 'Confirmed',names='Country/Region', height=600)

pie_chart_last20.update_traces(textposition='inside', textinfo='percent+label')



pie_chart_last20.update_layout(

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))



pie_chart_last20.show()
bar_data = corona_data.groupby(['Country/Region', 'ObservationDate'])['Confirmed', 'Deaths', 'Recovered'].sum().reset_index().sort_values('ObservationDate', ascending=True)

bar_fig = px.bar(bar_data, x="ObservationDate", y="Confirmed", color='Country/Region', text = 'Confirmed', orientation='v', height=1300,width=1000,

             title='Increase in COVID-19 Cases')

bar_fig.show()
bar_fig2 = px.bar(bar_data, x="ObservationDate", y="Deaths", color='Country/Region', text = 'Deaths', orientation='v', height=1000,width=900,

             title='COVID-19 Deaths since February to April 11th')

bar_fig2.show()
bar_fig3 = px.bar(bar_data, x="ObservationDate", y="Recovered", color='Country/Region', text = 'Recovered', orientation='v', height=1000,width=900,

             title='COVID-19 Recovered Cases since February to April 11th')

bar_fig3.show()
line_data = corona_data.groupby('ObservationDate').sum().reset_index()



line_data = line_data.melt(id_vars='ObservationDate', 

                 value_vars=['Confirmed', 

                             'Recovered', 

                             'Deaths'], 

                 var_name='Ratio', 

                 value_name='Value')



line_fig = px.line(line_data, x="ObservationDate", y="Value", line_shape="spline",color='Ratio', 

              title='Confirmed cases, Recovered cases, and Death Over Time')

line_fig.show()