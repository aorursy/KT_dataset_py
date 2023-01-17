import pandas as pd

import numpy as np

import plotly



import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()
#let's set ObservationDate as the index and drop the unneeded columns

data.index=data['ObservationDate']

data = data.drop(['SNo','ObservationDate'],axis=1)

data.head()
data_NZ = data[data['Country/Region']=='New Zealand']

data_NZ.tail()
data_NZ['daily_confirmed'] = data_NZ['Confirmed'].diff()

data_NZ['daily_deaths'] = data_NZ['Deaths'].diff()

data_NZ['daily_recovered'] = data_NZ['Recovered'].diff()
#Ranking by 'Confirmed' case

latest = data[data.index=='06/02/2020']

latest = latest.groupby('Country/Region').sum()

latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 



#New Zealand's Ranking

print('Ranking of New Zealand: ', latest[latest['Country/Region']=='New Zealand'].index.values[0]+1)
date = data_NZ[data_NZ['daily_confirmed']==data_NZ['daily_confirmed'].max()].index.values[0]

max_case = int(data_NZ['daily_confirmed'].max())

print('The max number of daily confirmed case is ',max_case,' which happened on ',date)
#Quick Viz

data_NZ['daily_confirmed'].plot()

data_NZ['daily_deaths'].plot()

data_NZ['daily_recovered'].plot()

plt.show()
from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=data_NZ.index, y=data_NZ['daily_confirmed'].values,name='Daily Confirmed')

daily_deaths_object = go.Scatter(x=data_NZ.index, y=data_NZ['daily_deaths'].values,name='Daily Deaths')

daily_recovered_object = go.Scatter(x=data_NZ.index, y=data_NZ['daily_recovered'].values,name='Daily Recovered')



#create a layout object

layout = go.Layout(title='New Zealand Daily cases 17B00107',xaxis=dict(title='Date'),yaxis=dict(title='Number of People'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout)



#add vertical line for lockdown

# Create scatter trace of text labels

fig.add_trace(go.Scatter(

    x=['04/03/2020','03/17/2020','03/23/2020'], 

    y=[100,90,80],

    text=["Level 4 (Nationwide Lockdown)","Alert Level 2 ","Level 3"],

    mode="text",

))





fig.add_shape(

        dict(

            type="line",

            x0='03/21/2020',

            y0=-10,

            x1='03/21/2020',

            y1=110,

            line=dict(

                color="green",

                width=1

            )

))



fig.add_shape(

        dict(

            type="line",

            x0='03/23/2020',

            y0=-10,

            x1='03/23/2020',

            y1=110,

            line=dict(

                color="magenta",

                width=1

            )

))



fig.add_shape(

        dict(

            type="line",

            x0='03/25/2020',

            y0=-10,

            x1='03/25/2020',

            y1=110,

            line=dict(

                color="Red",

                width=1

            )

))



fig.add_shape(

        dict(

            type="line",

            x0='04/27/2020',

            y0=-10,

            x1='04/27/2020',

            y1=110,

            line=dict(

                color="magenta",

                width=1

            )

))



fig.add_shape(

        dict(

            type="line",

            x0='05/13/2020',

            y0=-10,

            x1='05/13/2020',

            y1=110,

            line=dict(

                color="green",

                width=1

            )

))





iplot(fig)

fig.write_html('NewZealand_daily_cases_17B00107.html')
df1 = data_NZ[['daily_confirmed','daily_deaths','daily_recovered']]

df1 = df1.fillna(0)

styled_object = df1.style.background_gradient(cmap='jet').highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

f = open('table_17B00107.html','w')

f.write(styled_object.render())