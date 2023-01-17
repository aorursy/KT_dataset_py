import pandas as pd

import numpy as np

import os

import random

import plotly.offline as offline

import plotly.plotly as py

import plotly.graph_objs as go

from scipy import stats

from sklearn.linear_model import LinearRegression

from scipy.stats import chi2_contingency



%matplotlib inline

offline.init_notebook_mode(connected=True)
jobs = pd.read_csv("../input/Jobs_ML_DS.csv")

print(jobs.shape)

display(jobs.head())
jobs.as_of_date = pd.to_datetime(jobs.as_of_date)

jobs.number_of_openings.value_counts()
### Change # of openings Nan to 1 and outliers to 1

jobs.loc[jobs.number_of_openings.isna(),'number_of_openings'] = 1

jobs.loc[jobs.number_of_openings == 100,'number_of_openings'] = 1

jobs.number_of_openings.value_counts()
### sort by time

jobs = jobs.sort_values(by='as_of_date', ascending=True).reset_index(drop=True)

jobs.head()
### jobs for DS and ML positions

DS = jobs[jobs.title.str.contains('Data Scientist')==True].reset_index(drop=True)

ML = jobs[jobs.title.str.contains('Machine Learning')==True].reset_index(drop=True)

print(DS.shape)

print(ML.shape)

display(DS.head())
###### Plot the time series job posting information

DS_total = DS.groupby('as_of_date').sum()

ML_total = ML.groupby('as_of_date').sum()

trace1 = go.Scatter(

    x = DS_total.index,

    y = DS_total.number_of_openings,

    name = 'Data Scientist'

)

trace2 = go.Scatter(

    x = ML_total.index,

    y = ML_total.number_of_openings,

    name = 'Machine Learning Engineer'

)

layout = go.Layout(

        title = "Number of Openings",

        xaxis = dict(title='Day'),

        yaxis = dict(title="Num"),

)

data = [trace1, trace2]

fig = go.Figure(data=data, layout=layout)

offline.iplot(fig, show_link=False)
### Count the State information

jobs['State'] = jobs.region

jobs['Type'] = 'ML'

jobs.loc[jobs.region.isin(['CA', 'Menlo Park', 'California', 'CA,California']), 'State'] = 'CA'

jobs.loc[jobs.region.isin(['WA', 'Seattle']), 'State'] = 'WA'

jobs.loc[jobs.region.isin(['VA', 'Virginia']), 'State'] = 'VA'

jobs.loc[jobs.region.isin(['MA', 'MA,Mass']), 'State'] = 'MA'

jobs.loc[jobs.title.str.contains('Data Scientist'), 'Type'] = 'DS'

### Top 10 states

jobs.State.value_counts()[0:10]
job_state = jobs.groupby(['as_of_date', 'State']).sum()

job_state.head()
##### Plotting number of openings by top states

trace1 = go.Scatter(

    x = job_state.loc[pd.IndexSlice[:, 'CA'], :].index.get_level_values('as_of_date'),

    y = job_state.loc[pd.IndexSlice[:, 'CA'], :].number_of_openings,

    name = 'CA'

)

trace2 = go.Scatter(

    x = job_state.loc[pd.IndexSlice[:, 'WA'], :].index.get_level_values('as_of_date'),

    y = job_state.loc[pd.IndexSlice[:, 'WA'], :].number_of_openings,

    name = 'WA'

)

trace3 = go.Scatter(

    x = job_state.loc[pd.IndexSlice[:, 'MA'], :].index.get_level_values('as_of_date'),

    y = job_state.loc[pd.IndexSlice[:, 'MA'], :].number_of_openings,

    name = 'MA'

)

trace4 = go.Scatter(

    x = job_state.loc[pd.IndexSlice[:, 'VA'], :].index.get_level_values('as_of_date'),

    y = job_state.loc[pd.IndexSlice[:, 'VA'], :].number_of_openings,

    name = 'VA'

)

trace5 = go.Scatter(

    x = job_state.loc[pd.IndexSlice[:, 'TX'], :].index.get_level_values('as_of_date'),

    y = job_state.loc[pd.IndexSlice[:, 'TX'], :].number_of_openings,

    name = 'TX'

)

trace6 = go.Scatter(

    x = job_state.loc[pd.IndexSlice[:, 'IL'], :].index.get_level_values('as_of_date'),

    y = job_state.loc[pd.IndexSlice[:, 'IL'], :].number_of_openings,

    name = 'IL'

)

layout = go.Layout(

        title = "Num of Openings by States",

        xaxis = dict(title='Date'),

        yaxis = dict(title="Number"),

)

data = [trace1, trace2, trace3, trace4, trace5, trace6]

fig = go.Figure(data=data, layout=layout)

offline.iplot(fig, show_link=False)
jobs['Month_Y'] = jobs['as_of_date'].apply(lambda x: x.strftime('%m-%Y'))

GR_State = jobs.groupby(['Month_Y', 'State']).sum()

GR_Half = GR_State.loc[GR_State.index.get_level_values('State').isin(['CA', 'WA', 'MA', 'NY', 'VA', 

                                                                      'TX', 'IL', 'NJ', 'MD', 'PA',

                                                                     'CO', 'GA'])]

GR_Half_2017 = GR_Half.loc[GR_Half.index.get_level_values('Month_Y').isin(['01-2017', '02-2017', '03-2017', 

                                                                        '04-2017', '05-2017', '06-2017'])]

GR_Half_2018 = GR_Half.loc[GR_Half.index.get_level_values('Month_Y').isin(['01-2018', '02-2018', '03-2018', 

                                                                        '04-2018', '05-2018', '06-2018'])]

GR_Half_2018['GR']= (GR_Half_2018.number_of_openings.get_values() - GR_Half_2017.number_of_openings.

 get_values()) / GR_Half_2017.number_of_openings.get_values()
print(GR_Half_2018.GR.index.get_level_values('State')[0:12])

### Year-over-year results of top 12 growing states

GR_data = pd.DataFrame(np.asarray(GR_Half_2018.GR.get_values()).reshape(6,12))

GR_data.columns = GR_Half_2018.GR.index.get_level_values('State')[0:12]

display(GR_data)
###### Plot the growth rates by states

trace0 = go.Bar(

    x = GR_data.columns[0:12],

    y = GR_data.iloc[0,:].get_values(),

    name = 'Jan'

)

trace1 = go.Bar(

    x = GR_data.columns[0:12],

    y = GR_data.iloc[1,:].get_values(),

    name = 'Feb'

)

trace2 = go.Bar(

    x = GR_data.columns[0:12],

    y = GR_data.iloc[2,:].get_values(),

    name = 'Mar'

)

trace3 = go.Bar(

    x = GR_data.columns[0:12],

    y = GR_data.iloc[3,:].get_values(),

    name = 'Apr'

)

trace4 = go.Bar(

    x = GR_data.columns[0:12],

    y = GR_data.iloc[4,:].get_values(),

    name = 'May'

)

trace5 = go.Bar(

    x = GR_data.columns[0:12],

    y = GR_data.iloc[5,:].get_values(),

    name = 'June'

)

layout = go.Layout(

        title = "2017-2018 Year-over-Year Job Posting Growth Rate in Month by States",

        xaxis = dict(title='State'),

        yaxis = dict(title="Growth Rate (%/100)",

                     range=[-2, 18]),

)

data = [trace0, trace1, trace2, trace3, trace4, trace5]

fig = go.Figure(data=data, layout=layout)

offline.iplot(fig, show_link=False)
### Load Package

from fbprophet import Prophet
job_daily = jobs.groupby('as_of_date').sum().reset_index()

job_daily = job_daily.iloc[:,[0,2]]

display(job_daily.tail())
### Define Prediction Functions

def Prediction(data, train_end: str='2018-06-20', future_days: int=30):

    df = data

    df.columns = ['ds', 'y']

    training_time = train_end

    lag = future_days

    train_index = df.loc[(df.ds==str(training_time))].index.get_values()[0]

    df_train, df_test = df[0:train_index], df[train_index:(train_index+int(lag))]

    m = Prophet(holidays_prior_scale=0.5, seasonality_prior_scale=10, yearly_seasonality=True, interval_width=0.95)

#    m.add_seasonality(name='weekly', period=7, fourier_order=80, prior_scale=50)

    m.fit(df_train)

    future = m.make_future_dataframe(periods=lag, include_history=False)

    forecast = m.predict(future)

    ffcast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    ffcast = ffcast.set_index(ffcast.ds)

    df = df.set_index(df.ds)

    ffcast['Orig'] = df.y

    ffcast = ffcast.reset_index(drop=True)

    ffcast.columns = ['date', 'yhat', 'yhat_lower', 'yhat_upper', 'True_Value']

    return ffcast, df_train
ffcast, job_daily_orig = Prediction(job_daily, train_end='2018-06-20', future_days=26)
upper_bound = go.Scatter(

    x=ffcast['date'],

    y=ffcast['yhat_upper'],

    line = dict(

        color = "#444",

        width = 1),

    opacity=.5,

    showlegend=False)



trace = go.Scatter(

    name='Prediction',

    x=ffcast['date'],

    y=ffcast['yhat'],

    mode='lines',

    line = dict(

        width = 2))



trace1 = go.Scatter(

    name='True Volume',

    x=job_daily['ds'],

    y=job_daily['y'],

    mode='lines',

    line = dict(

        width = 1.5))



lower_bound = go.Scatter(

    x=ffcast['date'],

    y=ffcast['yhat_lower'],

    line = dict(

        color = "#444",

        width = 1),

    opacity=.5,

    name='prediction bound')





data = [upper_bound, lower_bound, trace, trace1]



layout = go.Layout(

    yaxis=dict(title='daily post volume'),

    title='Job Posting Volume Prediction with 95% C.I.',

    showlegend = True)



fig = go.Figure(data=data, layout=layout)

offline.iplot(fig, show_link=False)
ds = jobs[jobs.Type == 'DS']

ml = jobs[jobs.Type == 'ML']

ds_daily = ds.groupby('as_of_date').sum().reset_index()

ds_daily = ds_daily.iloc[:,[0,2]]

ml_daily = ml.groupby('as_of_date').sum().reset_index()

ml_daily = ml_daily.iloc[:,[0,2]]

dsfcast, ds_orig = Prediction(ds_daily, train_end='2018-06-20', future_days=26)

mlfcast, ml_orig = Prediction(ml_daily, train_end='2018-06-20', future_days=26)
### Only plotting the dates after 2018-01-01

ds_daily = ds_daily[ds_daily.ds >= '2018-01-01'].reset_index(drop=True)

ml_daily = ml_daily[ml_daily.ds >= '2018-01-01'].reset_index(drop=True)
### Plot the 95% prediction C.I with original posting volume and predicted volume

upper_bound_1 = go.Scatter(

    x=dsfcast['date'],

    y=dsfcast['yhat_upper'],

    line = dict(

        color = "#444",

        width = 1),

    opacity=.5,

    showlegend=False)

trace1 = go.Scatter(

    name='DS job posting prediction',

    x=dsfcast['date'],

    y=dsfcast['yhat'],

    mode='lines',

    line = dict(

        width = 2))

trace2 = go.Scatter(

    name='real DS job posting',

    x=ds_daily['ds'],

    y=ds_daily['y'],

    mode='lines',

    line = dict(

        width = 1.5))

lower_bound_1 = go.Scatter(

    x=dsfcast['date'],

    y=dsfcast['yhat_lower'],

    line = dict(

        color = "#444",

        width = 1),

    opacity=.5,

    name='95% prediction bound')

upper_bound_2 = go.Scatter(

    x=mlfcast['date'],

    y=mlfcast['yhat_upper'],

    line = dict(

        color = "#444",

        width = 1),

    opacity=.5,

    showlegend=False)

trace3 = go.Scatter(

    name='ML job posting prediction',

    x=mlfcast['date'],

    y=mlfcast['yhat'],

    mode='lines',

    line = dict(

        width = 2,

        color = 'rgb(145,191,219)'))

trace4 = go.Scatter(

    name='real ML job posting',

    x=ml_daily['ds'],

    y=ml_daily['y'],

    mode='lines',

    line = dict(

        width = 1.5,

        color = 'rgb(252.0, 141.0, 89.0)'))

lower_bound_2 = go.Scatter(

    x=mlfcast['date'],

    y=mlfcast['yhat_lower'],

    line = dict(

        color = "#444",

        width = 1),

    opacity=.5,

    showlegend=False)

data = [upper_bound_1, lower_bound_1, trace1, trace2, upper_bound_2, lower_bound_2, trace3, trace4]

layout = go.Layout(

    yaxis=dict(title='daily post volume'),

    title='Job Posting Historical Volume and Prediction',

    showlegend = True)

fig = go.Figure(data=data, layout=layout)

offline.iplot(fig, show_link=False)