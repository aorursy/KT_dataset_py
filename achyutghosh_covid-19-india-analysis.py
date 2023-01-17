# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as graph #for creating interactive graphs

import plotly.express as px

from plotly.subplots import make_subplots

from datetime import datetime # for date time columns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import main data set

covid_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')



#Converting date column into correct format

covid_df['ObservationDate']=pd.to_datetime(covid_df['ObservationDate'])



#Geting latest timestamp

latest_timestamp=covid_df.iloc[covid_df.last_valid_index()]['ObservationDate']

#Geting latest date

latest_date=latest_timestamp.strftime("%Y-%m-%d")

print("Data available till {}".format(latest_date))



#Checking Columns in data set

covid_df.head()
# Calculating active cases and added in covid_df dataframe

covid_df['ActiveCases'] = covid_df['Confirmed'] - covid_df['Deaths'] - covid_df['Recovered']

covid_df.head()
india_df = covid_df[covid_df['Country/Region'] == 'India']

india_df.head()
# Group by date

india_df = india_df.groupby(['ObservationDate']).sum().reset_index()

india_df.head()
# Visualizing number of Confirmed cases in India

fig_confirmed = px.line(india_df,x='ObservationDate',y='Confirmed',title='Confirmed Cases in India')

fig_confirmed.show()



# Visualizing number of Recovered cases in India

fig_recovered = px.line(india_df,x='ObservationDate',y='Recovered', title = 'Recovered cases in India')

fig_recovered.show()



# Visualizing number of Death cases in India

fig_recovered = px.line(india_df,x='ObservationDate',y='Deaths', title = 'Death cases in India')

fig_recovered.show()
#Visualizing Cumaltive trends

fig = graph.Figure()

fig.add_trace(graph.Scatter(x=india_df['ObservationDate'], y=india_df['Confirmed'],

                    mode='lines',name=' Confirmed Cases'))

fig.add_trace(graph.Scatter(x=india_df['ObservationDate'], y=india_df['Deaths'], 

                mode='lines',name='Deaths'))

fig.add_trace(graph.Scatter(x=india_df['ObservationDate'], y=india_df['Recovered'], 

                mode='lines',name='Recovered Cases'))

fig.add_trace(graph.Scatter(x=india_df['ObservationDate'], y=india_df['ActiveCases'], 

                mode='lines',name='Active Cases'))



        

    

fig.update_layout(title_text='Trend in India',plot_bgcolor='rgb(250, 242, 242)')



fig.show()