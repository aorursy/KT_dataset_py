# Project: Novel Corona Virus 2019 Dataset by Kaggle

# Program: COVID-19 in Canada with model

# Author:  Michel LeBlond

# Date:   April 3, 2020



# TODO explore ontario data and other models.







import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



import os



# Input data files are available in the "../input/" directory.



df_covid = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df_covid.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country', 'Province/State':'Province' }, inplace=True)

ondate=max(df_covid['Date'])

print ("The last observation date is " + ondate)
def plot_bar_chart(confirmed, deaths, recoverd, country, fig=None):

    if fig is None:

        fig = go.Figure()

    fig.add_trace(go.Bar(x=confirmed['Date'],

                y=confirmed['Confirmed'],

                name='Confirmed'

                ))

    fig.add_trace(go.Bar(x=deaths['Date'],

                y=deaths['Deaths'],

                name='Deaths'

                ))

    fig.add_trace(go.Bar(x=recovered['Date'],

                y=recovered['Recovered'],

                name='Recovered'

                ))



    fig.update_layout(

        title= 'Cumulative Daily Cases of COVID-19 (Confirmed, Deaths, Recovered) - ' + country + ' as of ' + ondate ,

        xaxis_tickfont_size=12,

        yaxis=dict(

            title='Number of Cases',

            titlefont_size=14,

            tickfont_size=12,

        ),

        legend=dict(

            x=0,

            y=1.0,

            bgcolor='rgba(255, 255, 255, 0)',

            bordercolor='rgba(255, 255, 255, 0)'

        ),

        barmode='group',

        bargap=0.15, 

        bargroupgap=0.1 

    )

    return fig
def plot_line_chart(confirmed, deaths, recoverd, country, fig=None):

    if fig is None:

        fig = go.Figure()

    fig.add_trace(go.Scatter(x=confirmed['Date'], 

                         y=confirmed['Confirmed'],

                         mode='lines+markers',

                         name='Confirmed'

                         ))

    fig.add_trace(go.Scatter(x=deaths['Date'], 

                         y=deaths['Deaths'],

                         mode='lines+markers',

                         name='Deaths'

                         ))

    fig.add_trace(go.Scatter(x=recovered['Date'], 

                         y=recovered['Recovered'],

                         mode='lines+markers',

                         name='Recovered'

                        ))

    fig.update_layout(

        title= 'Number of COVID-19 Cases Over Time - ' + country + ' as of ' + ondate ,

        xaxis_tickfont_size=12,

        yaxis=dict(

           title='Number of Cases',

           titlefont_size=14,

           tickfont_size=12,

        ),

        legend=dict(

           x=0,

           y=1.0,

           bgcolor='rgba(255, 255, 255, 0)',

           bordercolor='rgba(255, 255, 255, 0)'

        )

     )

    return fig
confirmed = df_covid.groupby('Date').sum()['Confirmed'].reset_index() 

deaths = df_covid.groupby('Date').sum()['Deaths'].reset_index() 

recovered = df_covid.groupby('Date').sum()['Recovered'].reset_index()
plot_bar_chart(confirmed, deaths, recovered,'Worldwide').show()
plot_line_chart(confirmed, deaths, recovered,'Worldwide').show()
full_table = df_covid.copy() # we need this to build the model later

# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']







full_table['Date'] = pd.to_datetime(full_table['Date'])

#full_table.tail()



# Clean Data for Canada 

full_table = full_table.replace(to_replace =["Calgary, Alberta", "Edmonton, Alberta"],value ="Alberta")

full_table =full_table.replace(to_replace =[" Montreal, QC"],value ="Quebec") 

# This generated duplicate dates that broke when running predictions through our model

# Changing to other labels for now.  Merge data on duplicate dates in next pass at code.

#full_table = full_table.replace(to_replace =["Toronto, ON", "London, ON"],value ="Ontario") 

#full_table =full_table.replace(to_replace =["Diamond Princess cruise ship"],value ="Ontario") 

full_table = full_table.replace(to_replace =["Toronto, ON", "London, ON"],value ="OntarioOther") 

full_table =full_table.replace(to_replace =["Diamond Princess cruise ship"],value ="OntarioOther") 



China_df = full_table[full_table['Country'] == 'Mainland China'].copy()

#China_df.tail() 

#full_table.Active.describe()



China_df.tail() 

#df_covid['Country'].unique()
def get_time_series(country):

    # for some countries, data is spread over several Provinces

    if full_table[full_table['Country'] == country]['Province'].nunique() > 1:

        country_table = full_table[full_table['Country'] == country]

        country_df = pd.DataFrame(pd.pivot_table(country_table, values = ['Confirmed', 'Deaths', 'Recovered', 'Active'],

                              index='Date', aggfunc=sum).to_records())

        return country_df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']]

    df = full_table[(full_table['Country'] == country) 

                & (full_table['Province'].isin(['', country]))]

    return df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']]





def get_time_series_province(province):

    # for some countries, data is spread over several Provinces

    df = full_table[(full_table['Province'] == province)]

    return df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']]
country = 'Mainland China'

df = get_time_series(country)

if len(df) > 1 and df.iloc[-2,0] >= df.iloc[-1,0]:

    df.drop(df.tail(1).index,inplace=True)

df.tail(10)
import math

def model_with_lag(N, a, alpha, lag, t):

    # we enforce N, a and alpha to be positive numbers using min and max functions

    lag = min(max(lag, -100), 100) # lag must be less than +/- 100 days 

    return max(N, 0) * (1 - math.e ** (min(-a, 0) * (t - lag))) ** max(alpha, 0)



def model(N, a, alpha, t):

    return max(N, 0) * (1 - math.e ** (min(-a, 0) * t)) ** max(alpha, 0)
model_index = 0



def model_loss(params):

#     N, a, alpha, lag = params

    N, a, alpha = params

    model_x = []

    r = 0

    for t in range(len(df)):

        r += (model(N, a, alpha, t) - df.iloc[t, model_index]) ** 2

#         r += (math.log(1 + model(N, a, alpha, t)) - math.log(1 + df.iloc[t, 0])) ** 2 

#         r += (model_with_lag(N, a, alpha, lag, t) - df.iloc[t, 0]) ** 2

#         print(model(N, a, alpha, t), df.iloc[t, 0])

    return math.sqrt(r) 
import numpy as np

from scipy.optimize import minimize

use_lag_model = False

if use_lag_model:

    opt = minimize(model_loss, x0=np.array([200000, 0.05, 15, 0]), method='Nelder-Mead', tol=1e-5).x

else:

    model_index = 0

    opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

    model_index = 1

    opt_deaths = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

    model_index = 2

    opt_recovered = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x
import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline 



model_x = []

for t in range(len(df)):

    model_x.append([df.index[t], model(*opt_confirmed, t), model(*opt_deaths, t), model(*opt_recovered, t)])

model_sim = pd.DataFrame(model_x, dtype=int)

model_sim.set_index(0, inplace=True)

model_sim.columns = ['Model-Confirmed', 'Model-Deaths', 'Model-Recovered']



model_sim['Model-Active'] = model_sim['Model-Confirmed'] - model_sim['Model-Deaths'] - model_sim['Model-Recovered']

model_sim.loc[model_sim['Model-Active']<0,'Model-Active'] = 0

plot_color = ['#99990077', '#FF000055', '#0000FF55', '#00FF0055', '#999900FF', '#FF0000FF', '#0000FFFF', '#00FF00FF']



pd.concat([model_sim, df], axis=1).plot(color = plot_color)

plt.show()
import datetime

start_date = df.index[0]

n_days = len(df) + 30

extended_model_x = []

last_row = []



isValid = True

last_death_rate = 0



for t in range(n_days):

    extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt_confirmed, t), model(*opt_deaths, t), model(*opt_recovered, t)])

   

    #if deaths + recovered > confirmed or deaths rate > 5%, maybe not valid

    if (t > len(df)):

        last_row = extended_model_x[-1]

        if (last_row[2] + last_row[3] > last_row[1]) or (last_row[2] > last_row[1]*0.12):

            if (isValid):

                last_row2 = extended_model_x[-2]

                last_death_rate = last_row2[2]/last_row2[1]

                isValid = False



        if (last_row[2] > last_row[1]*0.05):

            last_row[2] = last_row[1]*last_death_rate

            

        if (last_row[2] + last_row[3] > last_row[1]):

            last_row[2] = last_row[1]*last_death_rate

            last_row[3] = last_row[1]*(1-last_death_rate)



extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)

extended_model_sim.set_index(0, inplace=True)



extended_model_sim.columns = ['Model-Confirmed', 'Model-Deaths', 'Model-Recovered']

extended_model_sim['Model-Active'] = extended_model_sim['Model-Confirmed'] - extended_model_sim['Model-Deaths'] - extended_model_sim['Model-Recovered']

extended_model_sim.loc[extended_model_sim['Model-Active']<0,'Model-Active'] = 0



plot_color = ['#99990077', '#FF000055', '#0000FF55', '#00FF0055', '#999900FF', '#FF0000FF', '#0000FFFF', '#00FF00FF']



pd.concat([extended_model_sim, df], axis=1).plot(color = plot_color)

print('China COVID-19 Prediction')

plt.show()
df.tail()
pd.options.display.float_format = '{:20,.0f}'.format

concat_df = pd.concat([df, extended_model_sim], axis=1)

concat_df[concat_df.index.day % 3 == 0]
def display_fit(df, opt_confirmed, opt_deaths, opt_recovered, ax):

    model_x = []

    

    isValid = True

    last_death_rate = 0

    

    for t in range(len(df)):

        model_x.append([df.index[t], model(*opt_confirmed, t), model(*opt_deaths, t), model(*opt_recovered, t)])

        

        #if deaths + recovered > confirmed or deaths rate > 5%, maybe not valid

        if (t > len(df)):

            last_row = model_x[-1]

            if (last_row[2] + last_row[3] > last_row[1]) or (last_row[2] > last_row[1]*0.05):

                if (isValid):

                    last_row2 = model_x[-2]

                    last_death_rate = last_row2[2]/last_row2[1]

                    isValid = False

                    

            if (last_row[2] > last_row[1]*0.05):

                last_row[2] = last_row[1]*last_death_rate

                

            if (last_row[2] + last_row[3] > last_row[1]):

                last_row[2] = last_row[1]*last_death_rate

                last_row[3] = last_row[1]*(1-last_death_rate)

                

                

    model_sim = pd.DataFrame(model_x, dtype=int)

    model_sim.set_index(0, inplace=True)

    model_sim.columns = ['Model-Confirmed', 'Model-Deaths', 'Model-Recovered']



    model_sim['Model-Active'] = model_sim['Model-Confirmed'] - model_sim['Model-Deaths'] - model_sim['Model-Recovered']

    model_sim.loc[model_sim['Model-Active']<0,'Model-Active'] = 0

    plot_color = ['#99990077', '#FF000055', '#0000FF55', '#00FF0055', '#999900FF', '#FF0000FF', '#0000FFFF', '#00FF00FF']



    return pd.concat([model_sim, df], axis=1).plot(ax=ax, figsize=(14, 10), color = plot_color)



def display_extended_curve(df, opt_confirmed, opt_deaths, opt_recovered, ax):

    start_date = df.index[0]

    n_days = len(df) + 40

    extended_model_x = []

    

    isValid = True

    last_death_rate = 0

    

    for t in range(n_days):

        extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt_confirmed, t), model(*opt_deaths, t), model(*opt_recovered, t)])

        

        #if deaths + recovered > confirmed or deaths rate > 5%, maybe not valid

        if (t > len(df)):

            last_row = extended_model_x[-1]

            if (last_row[2] + last_row[3] > last_row[1]) or (last_row[2] > last_row[1]*0.05):

                if (isValid):

                    last_row2 = extended_model_x[-2]

                    last_death_rate = last_row2[2]/last_row2[1]

                    isValid = False

            

            if (last_row[2] > last_row[1]*0.05):

                last_row[2] = last_row[1]*last_death_rate

                    

            if (last_row[2] + last_row[3] > last_row[1]):

                last_row[2] = last_row[1]*last_death_rate

                last_row[3] = last_row[1]*(1-last_death_rate)

                

                

    extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)

    extended_model_sim.set_index(0, inplace=True)

    extended_model_sim.columns = ['Model-Confirmed', 'Model-Deaths', 'Model-Recovered']



    extended_model_sim['Model-Active'] = extended_model_sim['Model-Confirmed'] - extended_model_sim['Model-Deaths'] - extended_model_sim['Model-Recovered']

    

    extended_model_sim.loc[extended_model_sim['Model-Active']<0,'Model-Active'] = 0

    plot_color = ['#99990077', '#FF000055', '#0000FF55', '#00FF0055', '#999900FF', '#FF0000FF', '#0000FFFF', '#00FF00FF']



    return pd.concat([extended_model_sim, df], axis=1).plot(ax=ax, figsize=(14, 10), color = plot_color)





def opt_display_model(df, stats):

    # if the last data point repeats the previous one, or is lower, drop it

    if len(df) > 1 and df.iloc[-2,0] >= df.iloc[-1,0]:

        df.drop(df.tail(1).index,inplace=True)

    global model_index

    model_index = 0

    opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

    model_index = 1

    opt_deaths = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

    model_index = 2

    opt_recovered = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

    if min(opt_confirmed) > 0:

        stats.append([country, *opt_confirmed, *opt_deaths, *opt_recovered])

        n_plot = len(stats)

        plt.figure(1)

        ax1 = plt.subplot(221)

        display_fit(df, opt_confirmed, opt_deaths, opt_recovered, ax1)

        ax2 = plt.subplot(222)

        display_extended_curve(df, opt_confirmed, opt_deaths, opt_recovered, ax2)

        plt.show()
stats = []



df = full_table[['Province','Country', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']].groupby('Date').sum()

print('World COVID-19 Prediction (With China Data)')

opt_display_model(df, stats)
stats = []



df = full_table[full_table['Country'] != 'Mainland China'][['Province','Country', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']].groupby('Date').sum()

print('World COVID-19 Prediction(Without China Data)')

opt_display_model(df, stats)
stats = []



# Province Specify

for Province in ['Hong Kong', 'Hubei']:

    df = get_time_series_province(Province)

    print('{} COVID-19 Prediction'.format(Province))

    opt_display_model(df, stats)
stats = []



df = full_table[full_table['Country'] == 'Canada'][['Province','Country', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']].groupby('Date').sum()

print('World COVID-19 Prediction(Canada)')

opt_display_model(df, stats)
stats = []



# Province Specify

for Province in ['British Columbia','Ontario','Quebec']:

    df = get_time_series_province(Province)

    print('{} COVID-19 Prediction'.format(Province))

    opt_display_model(df, stats)
Canada_df = full_table[full_table['Country'] == 'Canada'].copy()

Ontario_df = Canada_df[Canada_df['Province'] == 'Ontario'].copy()





# tests to see why ontario is  not running in model.

# Some Active showed - 1

#Canada_df.head() 

#Canada_df['Province'].unique()

#Ontario_df.describe()

#Ontario_df[Ontario_df.Active < 0] = 0  move this up to the complete data_frame



#Ontario_df.describe()

#Export the ontario data to check integrity.

Ontario_df.to_csv('Ontario.csv', index=False)





Canada_df.tail()
confirmed = Canada_df.groupby(['Date', 'Province'])['Confirmed'].sum().reset_index()

provinces = Canada_df['Province'].unique()

provinces
# Clean Data

Canada_df = Canada_df.replace(to_replace =["Toronto, ON", "London, ON"],  

                            value ="Ontario") 

Canada_df = Canada_df.replace(to_replace =["Calgary, Alberta", "Edmonton, Alberta"],  

                            value ="Alberta") 

Canada_df =Canada_df.replace(to_replace =[" Montreal, QC"],  

                            value ="Quebec") 

# Here recovered is assumed to be from BC, and Cruise ship data from Ontario

#Canada_df =Canada_df.replace(to_replace =["Recovered"],  

#                            value ="British Columbia") 

Canada_df =Canada_df.replace(to_replace =["Diamond Princess cruise ship"],  

                            value ="Ontario") 
confirmed = Canada_df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = Canada_df.groupby('Date').sum()['Deaths'].reset_index()

recovered = Canada_df.groupby('Date').sum()['Recovered'].reset_index()
plot_bar_chart(confirmed, deaths, recovered,'Canada').show()
plot_line_chart(confirmed, deaths, recovered,'Canada').show()
provinces = Canada_df['Province'].unique()

#df = Canada_df

provinces
confirmed = Canada_df.groupby(['Date', 'Province'])['Confirmed'].sum().reset_index()
fig = go.Figure()

for province in provinces:

 

    fig.add_trace(go.Scatter(

        x=confirmed[confirmed['Province']==province]['Date'],

        y=confirmed[confirmed['Province']==province]['Confirmed'],

        name = province, # Style name/legend entry with html tags

        connectgaps=True # override default to connect the gaps

    ))

fig.update_layout(title="Number of Confirmed COVID-19 Cases Over Time - Canada - By Province" + ' as of ' + ondate)       

fig.show()
grouped_country = Canada_df.groupby(["Province"] ,as_index=False)["Confirmed","Recovered","Deaths"].last().sort_values(by="Confirmed",ascending=False)

grouped_country
fig = go.Figure()



fig.add_trace(go.Bar(

    

    y=grouped_country['Province'],

    x=grouped_country['Confirmed'],

    orientation='h',

    text=grouped_country['Confirmed']

    ))

fig.update_traces(textposition='outside')

fig.update_layout(title="Cumulative Number of COVID-19 Confirmed Cases - By Province" + ' as of ' + ondate)    

fig.show()
fig = go.Figure()



trace1 = go.Bar(

    x=grouped_country['Confirmed'],

    y=grouped_country['Province'],

    orientation='h',

    name='Confirmed'

)

trace2 = go.Bar(

    x=grouped_country['Deaths'],

    y=grouped_country['Province'],

    orientation='h',

    name='Deaths'

)

trace3 = go.Bar(

    x=grouped_country['Recovered'],

    y=grouped_country['Province'],

    orientation='h',

    name='Recovered'

)



data = [trace1, trace2, trace3]

layout = go.Layout(

    barmode='stack'

)



fig = go.Figure(data=data, layout=layout)

fig.update_layout(title="Stacked Number of COVID-19 Cases (Confirmed, Deaths, Recoveries) - Canada by Province" + ' as of ' + ondate)    

fig.show()
ts_confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

ts_confirmed.rename(columns={'Country/Region':'Country', 'Province/State':'Province' }, inplace=True)

Canada_C_ts = ts_confirmed[ts_confirmed['Country'] == 'Canada'].copy()

Canada_C_ts

ts_diff =Canada_C_ts[Canada_C_ts.columns[4:Canada_C_ts.shape[1]]]

new = ts_diff.diff(axis = 1, periods = 1) 

ynew=list(new.sum(axis=0))
fig = go.Figure()

fig.add_trace(go.Bar(

    y=ynew,

    x=ts_diff.columns,

    text=list(new.sum(axis=0)),

    ))

fig.update_traces(textposition='outside')

fig.update_layout(title="Epidemic Curve - Daily Number of COVID-19 Confirmed Cases in Canada " + ' as of  ' + ondate,

                 yaxis=dict(title='Number of Cases'))    

fig.show()