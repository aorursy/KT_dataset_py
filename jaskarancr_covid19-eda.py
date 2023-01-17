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

#init_notebook_mode{connected=True}

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

                    name='New_cases',line=dict(color='red')))

fig.add_trace(go.Scatter(x=Date_df['Date'], y=Date_df['New_recover'],

                    mode='lines+markers',

                    name='New_recover',line=dict(color='green')))

fig.update_xaxes(tick0=4, dtick=4)

fig.show()
Date_df['Closed_cases']=Date_df['Recovered']+Date_df['Deaths']
Date_df['Recover_percent']=(Date_df['Recovered']/Date_df['Closed_cases'])*100

Date_df['Death_percent']=(Date_df['Deaths']/Date_df['Closed_cases'])*100

Date_df.head(5)

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

fig.update_layout(height=500, width=1000, title_text="Case progrssion of top countries")

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



fig.update_layout(height=500, width=1000, title_text="Death progrssion of top countries")

fig.show()

train_dataset = pd.read_csv('../input/time-covid-19/time_series_covid_19_confirmed.csv')

drop_clo = ['Province/State','Country/Region','Lat','Long']

train_dataset=train_dataset.drop(drop_clo,axis=1)

datewise= list(train_dataset.columns)

val_dataset = train_dataset[datewise[-15:]]
date_array=np.asarray(Date_df['Date'])

fig = make_subplots(rows=3, cols=1)



fig.add_trace(

    go.Scatter(x=date_array[:-15], mode='lines+markers', y=train_dataset.loc[0].values[:-15], marker=dict(color="dodgerblue"), showlegend=False,),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=date_array[-15:], y=val_dataset.loc[0].values, mode='lines+markers', marker=dict(color="darkorange"), showlegend=False,),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=date_array[:-15], mode='lines+markers', y=train_dataset.loc[1].values[:-15], marker=dict(color="dodgerblue"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=date_array[-15:], y=val_dataset.loc[1].values, mode='lines+markers', marker=dict(color="darkorange"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=date_array[:-15], mode='lines+markers', y=train_dataset.loc[2].values[:-15], marker=dict(color="dodgerblue"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=date_array[-15:], y=val_dataset.loc[2].values, mode='lines+markers', marker=dict(color="darkorange"), showlegend=False),

    row=3, col=1

)



fig.update_layout(height=1200, width=800, title_text="Train (blue) vs. Validation (orange) sales")

fig.show()
predictions = []

for i in range(len(val_dataset.columns)):

    if i == 0:

        predictions.append(train_dataset[train_dataset.columns[-16]].values)

    else:

        predictions.append(val_dataset[val_dataset.columns[i-1]].values)

    

predictions = np.transpose(np.array([row.tolist() for row in predictions]))

error_naive = np.linalg.norm(predictions[:] - val_dataset.values[:])/len(predictions[:])
pred_1 = predictions[0]

pred_2 = predictions[1]

pred_3 = predictions[2]



fig = make_subplots(rows=3, cols=1)



fig.add_trace(

    go.Scatter(x=date_array[:-15], mode='lines+markers', y=train_dataset.loc[0].values[:-15], marker=dict(color="dodgerblue"),name="Train"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=date_array[-15:], y=val_dataset.loc[0].values, mode='lines+markers', marker=dict(color="darkorange"), name="Validation"),

    row=1, col=1,

)



fig.add_trace(

    go.Scatter(x=date_array[-15:], y=pred_1, mode='lines', marker=dict(color="seagreen"),

               name="Pred"),

    row=1, col=1

)



fig.add_trace(

    go.Scatter(x=date_array[:-15], mode='lines+markers', y=train_dataset.loc[1].values[:-15], marker=dict(color="dodgerblue"), showlegend=False),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=date_array[-15:], y=val_dataset.loc[1].values, mode='lines+markers', marker=dict(color="darkorange"), showlegend=False),

    row=2, col=1

)





fig.add_trace(

    go.Scatter(x=date_array[-15:], y=pred_2, mode='lines', marker=dict(color="seagreen"), showlegend=False,

               name="Denoised signal"),

    row=2, col=1

)



fig.add_trace(

    go.Scatter(x=date_array[:-15], mode='lines+markers', y=train_dataset.loc[2].values[:-15], marker=dict(color="dodgerblue"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=date_array[-15:], y=val_dataset.loc[2].values, mode='lines+markers', marker=dict(color="darkorange"), showlegend=False),

    row=3, col=1

)



fig.add_trace(

    go.Scatter(x=date_array[-15:], y=pred_3, mode='lines', marker=dict(color="seagreen"), showlegend=False,

               name="Denoised signal"),

    row=3, col=1

)



fig.update_layout(height=1200, width=800, title_text="Naive approach")

fig.show()
from sklearn.metrics import mean_squared_error

from math import sqrt

rms = sqrt(mean_squared_error(predictions[:] ,val_dataset.values[:]))

print(rms)
model_train=Date_df.iloc[:int(Date_df.shape[0]*0.90)]

valid=Date_df.iloc[int(Date_df.shape[0]*0.90):]
import statsmodels.api as sm

from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing

holt=Holt(np.asarray(model_train["Confirmed"])).fit(smoothing_level=0.2, smoothing_slope=0.8)

y_pred=valid.copy()
import matplotlib.pyplot as plt

y_pred["Holt"]=holt.forecast(len(valid))

#model_scores.append(np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"])))

print("Root Mean Square Error Holt's Linear Model: ",np.sqrt(mean_squared_error(y_pred["Confirmed"],y_pred["Holt"])))
fig = go.Figure()

fig.add_trace(go.Scatter(x=Date_df['Date'], y=model_train['Confirmed'],

                    mode='lines+markers',

                    name='Train ',line=dict(color='green')))

fig.add_trace(go.Scatter(x=valid['Date'], y=valid['Confirmed'],

                    mode='lines+markers',

                    name='validation ',line=dict(color='red')))

fig.add_trace(go.Scatter(x=valid['Date'], y=y_pred["Holt"],

                    mode='lines+markers',

                    name='predicted ',line=dict(color='white')))

fig.update_xaxes(tick0=4, dtick=4)

fig.update_layout(title='Holt Linear Model',

                   xaxis_title='Date',

                   yaxis_title='Percentage')

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