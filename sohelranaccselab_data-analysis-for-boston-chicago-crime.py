# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

pd.set_option('display.max_rows', None)

import datetime

from plotly.subplots import make_subplots
data = pd.read_csv('../input/crimes-in-boston/crime.csv',encoding='latin')
data.head()
data.info()
def treemap(categories,title,path,values):

    fig = px.treemap(categories, path=path, values=values, height=700,

                 title=title, color_discrete_sequence = px.colors.sequential.RdBu)

    fig.data[0].textinfo = 'label+text+value'

    fig.show()
def histogram(data,path,color,title,xaxis,yaxis):

    fig = px.histogram(data, x=path,color=color)

    fig.update_layout(

        title_text=title,

        xaxis_title_text=xaxis, 

        yaxis_title_text=yaxis, 

        bargap=0.2, 

        bargroupgap=0.1

    )

    fig.show()
def bar(categories,x,y,color,title,xlab,ylab):

    fig = px.bar(categories, x=x, y=y,

             color=color,

             height=400)

    fig.update_layout(

    title_text=title, 

    xaxis_title_text=xlab, 

    yaxis_title_text=ylab,

    bargap=0.2, 

    bargroupgap=0.1

    )

    fig.show()

Number_crimes = data['OFFENSE_CODE_GROUP'].value_counts()

values = Number_crimes.values

categories = pd.DataFrame(data=Number_crimes.index, columns=["OFFENSE_CODE_GROUP"])

categories['values'] = values
treemap(categories,'Major Crimes in Boston',['OFFENSE_CODE_GROUP'],categories['values'])
histogram(data,"OFFENSE_CODE_GROUP","OFFENSE_CODE_GROUP",'Major Crimes in Boston','Crime','Count')
bar(categories,categories['OFFENSE_CODE_GROUP'][0:10],categories['values'][0:10]

    ,categories['OFFENSE_CODE_GROUP'][0:10],'Top 10 Major Crimes in Boston','Crime','Count')
Number_crimes_year = data['YEAR'].value_counts()

years = pd.DataFrame(data=Number_crimes_year.index, columns=["YEAR"])

years['values'] = Number_crimes_year.values


fig = px.pie(years, values='values', names='YEAR', color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
Number_crimes_month = data['MONTH'].value_counts()

months = pd.DataFrame(data=Number_crimes_month.index, columns=["MONTH"])

months['values'] = Number_crimes_month.values
fig = go.Figure(go.Bar(

            x=months['values'],

            y=months['MONTH'],

        marker=dict(

            color='rgb(13,143,129)',



        ),

            orientation='h'))

fig.update_layout(

    title_text='Major Crimes in Boston per month', 

    xaxis_title_text='Count',

    yaxis_title_text='Month', 

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
Number_crimes_days = data['DAY_OF_WEEK'].value_counts()

days = pd.DataFrame(data=Number_crimes_days.index, columns=["DAY_OF_WEEK"])

days['values'] = Number_crimes_days.values
fig = px.histogram(data, y="DAY_OF_WEEK",color="DAY_OF_WEEK")

fig.update_layout(

    title_text='Crime count on each day', 

    xaxis_title_text='Day',

    yaxis_title_text='Crimes Count', 

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()

fig = go.Figure(data=[go.Pie(labels=days['DAY_OF_WEEK'], values=days['values'], hole=.4)])

fig.update_layout(

    title_text='Crime count on each day',

)

fig.show()
histogram(data,"HOUR","HOUR",'Crime count on each Hour','Hour','Count')
histogram(data,"YEAR","MONTH",'Crime count on each year per month','Year','Crimes Count')
Number_crimes_street = data['STREET'].value_counts()

street = pd.DataFrame(data=Number_crimes_street.index, columns=["STREET"])

street['values'] = Number_crimes_street.values
bar(street,street['STREET'][0:10],street['values'][0:10]

    ,street['STREET'][0:10],'Top 10 Crime count on each Street','Street',' Crime Count')
histogram(data,"OFFENSE_CODE_GROUP","YEAR",'Crime count per Category on each Year','Category','Crimes Count on each Year')
histogram(data,"OFFENSE_CODE_GROUP","MONTH",'Crime count per Category on each Month','Category','Crimes Count on each Month')
histogram(data,"MONTH","DAY_OF_WEEK",'Crime count per Month on each Day','Month','Crimes Count on each Day')
histogram(data,"DAY_OF_WEEK","HOUR",'Crime count per Day on each Hour','Day','Crimes Count on each Hour')
Data_2018 = data [(data['YEAR'] == 2018) ].reset_index(drop=True)
Number_crimes_2018 = Data_2018['OFFENSE_CODE_GROUP'].value_counts()

categories_2018 = pd.DataFrame(data=Number_crimes_2018.index, columns=["OFFENSE_CODE_GROUP"])

categories_2018['values'] = Number_crimes_2018.values
treemap(categories_2018,'Major Crimes in Boston in 2018',['OFFENSE_CODE_GROUP'],categories_2018['values'])

histogram(Data_2018,"OFFENSE_CODE_GROUP","OFFENSE_CODE_GROUP",'Major Crimes in Boston in 2018','Crime','Count')
fig = px.bar(categories_2018, x=categories_2018['OFFENSE_CODE_GROUP'][0:10], y=categories_2018['values'][0:10],

             color=categories_2018['OFFENSE_CODE_GROUP'][0:10],

             labels={'pop':'population of Canada'}, height=400)

fig.update_layout(

    title_text='Top 10 Major Crimes in Boston in 2018', # title of plot

    xaxis_title_text='Crime', # xaxis label

    yaxis_title_text='Count', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
Number_crimes_month_2018 = Data_2018['MONTH'].value_counts()

months_2018 = pd.DataFrame(data=Number_crimes_month_2018.index, columns=["MONTH"])

months_2018['values'] = Number_crimes_month_2018.values
fig = go.Figure(go.Bar(

            y=months_2018['values'],

            x=months_2018['MONTH'],

        marker=dict(

            color='blue',



        ),

            orientation='v'))

fig.update_layout(

    title_text='Major Crimes in Boston per month in 2018', # title of plot

    xaxis_title_text='Month', # xaxis label

    yaxis_title_text='Count ', # yaxis label

    bargap=0.2, 

    bargroupgap=0.1

)

fig.show()
Number_crimes_days_2018= Data_2018['DAY_OF_WEEK'].value_counts()

days_2018= pd.DataFrame(data=Number_crimes_days_2018.index, columns=["DAY_OF_WEEK"])

days_2018['values'] = Number_crimes_days_2018.values
histogram(Data_2018,"DAY_OF_WEEK","DAY_OF_WEEK",'Crime count on each day in 2018','Day','Crimes Count')
fig = go.Figure(data=[go.Pie(labels=days_2018['DAY_OF_WEEK'], values=days_2018['values'])])

fig.update_layout(

    title_text='Crime count on each day in 2018', # title of plot

)

fig.show()
histogram(Data_2018,"OFFENSE_CODE_GROUP","MONTH",'Crime count per Category on each Month in 2018','Category','Crimes Count on each Month')
histogram(Data_2018,"MONTH","DAY_OF_WEEK",'Crime count per Month on each Day in 2018','Month','Crimes Count on each Day')
histogram(Data_2018,"DAY_OF_WEEK","HOUR",'Crime count per Day on each Hour in 2018','Day','Crimes Count on each Hour')
import bq_helper

from bq_helper import BigQueryHelper



chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="chicago_crime")
bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")

bq_assistant.list_tables()
bq_assistant.head("crime", num_rows=5)
bq_assistant.table_schema("crime")
query1 = """SELECT

  primary_type,

  description,

  COUNTIF(year = 2015) AS arrests_2015,

  COUNTIF(year = 2016) AS arrests_2016,

  FORMAT('%3.2f', (COUNTIF(year = 2016) - COUNTIF(year = 2015)) / COUNTIF(year = 2015)*100) AS pct_change_2015_to_2016

FROM

  `bigquery-public-data.chicago_crime.crime`

WHERE

  arrest = TRUE

  AND year IN (2015,

    2016)

GROUP BY

  primary_type,

  description

HAVING

  COUNTIF(year = 2015) > 100

ORDER BY

  (COUNTIF(year = 2016) - COUNTIF(year = 2015)) / COUNTIF(year = 2015) DESC

        """

response1 = chicago_crime.query_to_pandas_safe(query1)

response1.head(10)
query2 = """SELECT

  year,

  month,

  incidents

FROM (

  SELECT

    year,

    EXTRACT(MONTH

    FROM

      date) AS month,

    COUNT(1) AS incidents,

    RANK() OVER (PARTITION BY year ORDER BY COUNT(1) DESC) AS ranking

  FROM

    `bigquery-public-data.chicago_crime.crime`

  WHERE

    primary_type = 'MOTOR VEHICLE THEFT'

    AND year <= 2016

  GROUP BY

    year,

    month )

WHERE

  ranking = 1

ORDER BY

  year DESC

        """

response2 = chicago_crime.query_to_pandas_safe(query2)

response2.head(10)