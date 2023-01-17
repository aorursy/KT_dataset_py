# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import pandas as pd

%matplotlib inline

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots





import folium 

from folium import plugins

from tqdm.notebook import tqdm as tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cleaned_data=pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

cleaned_data.head(15)
# cases 

cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Active Case = confirmed - deaths - recovered

cleaned_data['Active'] = cleaned_data['Confirmed'] - cleaned_data['Deaths'] - cleaned_data['Recovered']



# filling missing values 

cleaned_data[['Province/State']] = cleaned_data[['Province/State']].fillna('')

cleaned_data[cases] = cleaned_data[cases].fillna(0)



cleaned_data.head()
Date_df=cleaned_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()

Date_df=Date_df.sort_values(by=['Confirmed'])

Date_df.head(40)
temp = Date_df.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='case', value_name='count')

fig = px.area(temp, x="Date", y="count", color='case',

             title='Cases over time: Area Plot', color_discrete_sequence = ['cyan', 'red', 'orange'])

fig.update_xaxes(tick0=4, dtick=4)



fig.show()
for column in Date_df[['Confirmed']]:

   # Select column contents by column name using [] operator

   columnSeriesObj = Date_df[column]

#    print('Colunm Name : ', column)

#    print('Column Contents : ', columnSeriesObj.values)

   new_cases=columnSeriesObj.values

my_list=[]

for i in range(0,len(new_cases)-1):

    new_value=new_cases[i+1]-new_cases[i]

    my_list.append(new_value)

my_list.insert(0,0)



print(len(my_list))

print(Date_df.shape)

Date_df['New_cases'] = np.array(my_list)

import plotly.express as px

fig = go.Figure()

fig.add_trace(go.Bar(

    x=Date_df['Date'],

    y=Date_df['New_cases'],

    name='New_cases',

    marker_color='grey'

))

fig.show()
for column in Date_df[['Recovered']]:

   # Select column contents by column name using [] operator

   columnSeriesObj = Date_df[column]

#    print('Colunm Name : ', column)

#    print('Column Contents : ', columnSeriesObj.values)

   new_recover=columnSeriesObj.values

my_list1=[]

for i in range(0,len(new_recover)-1):

    new_value=new_recover[i+1]-new_recover[i]

    my_list1.append(new_value)

my_list1.insert(0,0)

print(len(my_list1))

print(Date_df.shape)

Date_df['New_recover'] = np.array(my_list1)

Date_df.head(5)
fig = go.Figure()

fig.add_trace(go.Scatter(x=Date_df['Date'], y=Date_df['New_cases'],

                    mode='lines+markers',

                    name='New_cases'))

fig.add_trace(go.Scatter(x=Date_df['Date'], y=Date_df['New_recover'],

                    mode='lines+markers',

                    name='New_recover'))

fig.update_xaxes(tick0=4, dtick=4)

fig.show()
Date_df['Closed_cases']=Date_df['Recovered']+Date_df['Deaths']
Date_df['Recover_percent']=(Date_df['Recovered']/Date_df['Closed_cases'])*100

Date_df['Death_percent']=(Date_df['Deaths']/Date_df['Closed_cases'])*100

Date_df

fig = go.Figure()

fig.add_trace(go.Scatter(x=Date_df['Date'], y=Date_df['Recover_percent'],

                    mode='lines+markers',

                    name='Recover %',line=dict(color='green')))

fig.add_trace(go.Scatter(x=Date_df['Date'], y=Date_df['Death_percent'],

                    mode='lines+markers',

                    name='Death %',line=dict(color='red')))

fig.update_xaxes(tick0=4, dtick=4)

fig.update_layout(title='Output of Closed Cases(Recovery OR Death)',

                   xaxis_title='Date',

                   yaxis_title='Percentage')

fig.show()
new_df=cleaned_data.loc[cleaned_data['Country/Region'] == 'China'] 

# v=cleaned_data.groupby('Country/Region')['Active'].sum().reset_index()

# v = v.melt(id_vars="Country/Region", value_vars=['Active'],

#                  var_name='case', value_name='count')

new_df1=cleaned_data.loc[cleaned_data['Date'] == '4/7/20'] 

res_df=new_df1.groupby('Country/Region')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()

fig = px.pie(res_df, values='Confirmed', names='Country/Region', title='Countries Cases Distribution')

fig.show()
top_10=res_df.sort_values(by=['Confirmed'],ascending=False)[0:10]
import plotly.express as px

fig = px.bar(top_10, x='Country/Region', y='Confirmed')

fig.show()

top_10_death=res_df.sort_values(by=['Deaths'],ascending=False)[0:10]

top_10_death.head(5)

import plotly.express as px

fig = px.bar(top_10_death, x='Country/Region', y='Deaths')

fig.show()
us_data=cleaned_data.loc[cleaned_data['Country/Region']=='US']

us_data=us_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()

us_data=us_data.sort_values(by=['Confirmed'])





Spain_data=cleaned_data.loc[cleaned_data['Country/Region']=='Spain']

Spain_data=Spain_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()

Spain_data=Spain_data.sort_values(by=['Confirmed'])







Italy_data=cleaned_data.loc[cleaned_data['Country/Region']=='Italy']

Italy_data=Italy_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()

Italy_data=Italy_data.sort_values(by=['Confirmed'])







France_data=cleaned_data.loc[cleaned_data['Country/Region']=='France']

France_data=France_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()

France_data=France_data.sort_values(by=['Confirmed'])







Germany_data=cleaned_data.loc[cleaned_data['Country/Region']=='Germany']

Germany_data=Germany_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()

Germany_data=Germany_data.sort_values(by=['Confirmed'])





fig = go.Figure()

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=us_data['Confirmed'][30:],

                    mode='markers',

                    name='US'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Spain_data['Confirmed'][30:],

                    mode='markers',

                    name='Spain'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Italy_data['Confirmed'][30:],

                    mode='markers',

                    name='Italy'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=France_data['Confirmed'][30:],

                    mode='markers',

                    name='France'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Germany_data['Confirmed'][30:],

                    mode='markers',

                    name='Germany'))

fig.update_xaxes(tick0=4, dtick=4)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=us_data['Deaths'][30:],

                    mode='markers',

                    name='US'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Spain_data['Deaths'][30:],

                    mode='markers',

                    name='Spain'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Italy_data['Deaths'][30:],

                    mode='markers',

                    name='Italy'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=France_data['Deaths'][30:],

                    mode='markers',

                    name='France'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Germany_data['Deaths'][30:],

                    mode='markers',

                    name='Germany'))

fig.update_xaxes(tick0=4, dtick=4)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=us_data['Active'][30:],

                    mode='markers',

                    name='US'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Spain_data['Active'][30:],

                    mode='markers',

                    name='Spain'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Italy_data['Active'][30:],

                    mode='markers',

                    name='Italy'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=France_data['Active'][30:],

                    mode='markers',

                    name='France'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Germany_data['Active'][30:],

                    mode='markers',

                    name='Germany'))

fig.update_xaxes(tick0=4, dtick=4)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=us_data['Recovered'][30:],

                    mode='markers',

                    name='US'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Spain_data['Recovered'][30:],

                    mode='markers',

                    name='Spain'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Italy_data['Recovered'][30:],

                    mode='markers',

                    name='Italy'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=France_data['Recovered'][30:],

                    mode='markers',

                    name='France'))

fig.add_trace(go.Scatter(x=us_data['Date'][30:], y=Germany_data['Recovered'][30:],

                    mode='markers',

                    name='Germany'))

fig.update_xaxes(tick0=4, dtick=4)

fig.show()
us_data=cleaned_data.loc[cleaned_data['Country/Region']=='US']

us_data=us_data.groupby('Date')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()

us_data=us_data.sort_values(by=['Confirmed'])

state_data=cleaned_data.loc[cleaned_data['Country/Region']=='US']

state_data.head(5)
fig = go.Figure()

fig.add_trace(go.Scatter(x=us_data['Date'], y=us_data['Active'],

                    mode='lines+markers',

                    name='Active'))

fig.add_trace(go.Scatter(x=us_data['Date'], y=us_data['Deaths'],

                    mode='lines+markers',

                    name='Deaths'))

fig.update_xaxes(tick0=4, dtick=4)

fig.show()
new_df=cleaned_data.loc[cleaned_data['Country/Region'] == 'China'] 

# v=cleaned_data.groupby('Country/Region')['Active'].sum().reset_index()

# v = v.melt(id_vars="Country/Region", value_vars=['Active'],

#                  var_name='case', value_name='count')

new_df1=cleaned_data.loc[cleaned_data['Date'] == '4/7/20'] 

res_df=new_df1.groupby('Country/Region')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()
new_df1=cleaned_data.loc[cleaned_data['Date'] == '4/7/20'] 
res_df=new_df1.groupby('Country/Region')['Confirmed','Recovered', 'Deaths', 'Active'].sum().reset_index()


import plotly.graph_objects as go







fig = go.Figure()

fig.add_trace(go.Bar(

    x=top_10['Country/Region'],

    y=top_10['Confirmed'],

    name='Confirmed',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=top_10['Country/Region'],

    y=top_10['Active'],

    name='Active',

    marker_color='lightsalmon'

))



fig.add_trace(go.Bar(

    x=top_10['Country/Region'],

    y=top_10['Deaths'],

    name='Deaths',

    marker_color='red'

))

fig.add_trace(go.Bar(

    x=top_10['Country/Region'],

    y=top_10['Recovered'],

    name='Recovered',

    marker_color='green'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45)

fig.show()
import plotly.express as px

fig = px.bar(top_10, x="sex", y="total_bill", color="smoker", barmode="group",

             facet_row="time", facet_col="day",

             category_orders={"day": ["Thur", "Fri", "Sat", "Sun"],

                              "time": ["Lunch", "Dinner"]})

fig.show()