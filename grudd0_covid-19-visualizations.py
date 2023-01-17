# import packages 

import numpy as np

import pandas as pd

from scipy.interpolate import interp1d

import plotly.graph_objects as go
# read covin-19 dataset into pandas dataframe

# Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE

# original reference: https://github.com/CSSEGISandData/COVID-19

df_confirmed = pd.read_csv("../input/corona-virus-time-series-dataset/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")

# change 'US' to 'United States' to be consistant with key in population file

df_confirmed.loc[df_confirmed[df_confirmed['Country/Region']  == 'US'].index,\

                 'Country/Region'] = 'United States'

# sum levels on a country basis

df_confirmed_sum = df_confirmed.groupby(['Country/Region']).sum()

n = 10  # look at top 10 countries

df_confirmed_sum = df_confirmed_sum.sort_values(by=df_confirmed.columns[-1],\

                                                ascending=False)[0:n]

df_confirmed_sum = df_confirmed_sum.drop(['Lat','Long'],axis=1)
# calculate population in millions

# read country population data

scale_pop = 1000000  # scale population number by millions

population = []

df_pop = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

for j in range(n):

    country = df_confirmed_sum.index[j]

    population.append((df_pop[df_pop['Country (or dependency)'] == country]\

                       ['Population (2020)'] / scale_pop).to_numpy()[0])
countries=df_confirmed_sum.index

cumulative = df_confirmed_sum.iloc[:,-1]



fig = go.Figure(data=[

    go.Bar(name='Total Cases', x=countries, y=cumulative),

    go.Bar(name='Normalized', x=countries, y=cumulative / population )

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_yaxes(type='log')



fig.update_layout(title="Countries with most Covid Cases<br>Total Confirmed Cases and<br>Total Confirmed Cases Normalized per Million in Population",

                title_x=0.5,

                  yaxis_title="Cases - Cases Normalized")

fig.show()
# cumulative cases

x_vec = []

y_vec = []

scale = 10

# interpolate to find 10 per million point

for j in range(n):

    y = df_confirmed_sum.iloc[j,:].to_numpy()  

    x = range(len(y))

    f = interp1d(x,y)

    xi = np.linspace(0,x[-1],500)

    yi = f(xi)

    yin = yi / population[j]

    y_vec.append(yin[yin > scale])  # just look at data greater them 10 per million

    xin = xi[yin > scale]           # just look at data greater them 10 per million

    x_vec.append(xin - xin[0])

# plot cumulative cases

x_vec = []

y_vec = []

scale = 10

for j in range(n):

    y = df_confirmed_sum.iloc[j,:].to_numpy()  

    x = range(len(y))

    f = interp1d(x,y)

    xi = np.linspace(0,x[-1],500)

    yi = f(xi)

    yin = yi / population[j]

    y_vec.append(yin[yin > scale])  # just look at data greater them 10 per million

    xin = xi[yin > scale]   

    yi = yi[yin > scale] # just look at data greater them 10 per million

    x_vec.append(xin - xin[0])



# generate plotly interactive figure

fig = go.Figure()

for j in range(n):

    fig.add_trace((go.Scatter(x=x_vec[j],y=y_vec[j],mode='lines+markers',\

                              marker=dict(size=4),

                              name=df_confirmed_sum.index[j],\

                              text=df_confirmed_sum.index[j])))

    

fig.update_xaxes(range=[0, 80])

fig.update_yaxes(range=[1, 4.5])

fig.update_layout(title="Countries with most Covid Cases <br> Cases Normalized by Million in Population",

                  #xaxis_type="log",

                  yaxis_type="log",

                  xaxis_title="Days from 10 per Million in Population",

                                    title_x=0.5,

                  yaxis_title="Cases Normalized by Million in Population",

                  )
# read covin-19 dataset into pandas dataframe

# Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE

# https://github.com/CSSEGISandData/COVID-19

df_death = pd.read_csv("../input/corona-virus-time-series-dataset/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")

# change 'US' to 'United States'

df_death.loc[df_death[df_death['Country/Region']  == 'US'].index,\

             'Country/Region'] = 'United States'

# sum levels on a country basis

df_death_sum = df_death.groupby(['Country/Region']).sum()

n = 10  # look at top 10 countries

df_death_sum = df_death_sum.sort_values(by=df_death.columns[-1],ascending=False)[0:n]

df_death_sum = df_death_sum.drop(['Lat','Long'],axis=1)
# calculate population in millions

# read country population data

scale_pop = 1000000

population = []

df_pop = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

#for j in range(len(df_country_sorted_top)):

for j in range(n):

    country = df_death_sum.index[j]

    population.append((df_pop[df_pop['Country (or dependency)'] == country]\

                       ['Population (2020)'] / scale_pop).to_numpy()[0])
#import plotly.graph_objects as go

countries=df_death_sum.index

cumulative = df_death_sum.iloc[:,-1]



fig = go.Figure(data=[

    go.Bar(name='Total Deaths', x=countries, y=cumulative),

    go.Bar(name='Normalized', x=countries, y=cumulative / population )

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.update_yaxes(type='log')



fig.update_layout(title="Countries with most Deaths<br>Total Deaths and<br>Deaths Normalized by Million in Population",

                  title_x=0.5,

                  yaxis_title="Deaths - Deaths Normalized")

fig.show()
# cumulative cases

x_vec = []

y_vec = []

scale = 1

for j in range(n):

    y = df_death_sum.iloc[j,:].to_numpy()  

    x = range(len(y))

    f = interp1d(x,y)

    xi = np.linspace(0,x[-1],500)

    yi = f(xi)

# deaths

x_vec = []

y_vec = []

scale = 1

for j in range(n):

    y = df_death_sum.iloc[j,:].to_numpy()  

    x = range(len(y))

    f = interp1d(x,y)

    xi = np.linspace(0,x[-1],500)

    yi = f(xi)

    yin = yi / population[j]

    y_vec.append(yin[yin > scale]) # just look at data greater them 1 per million

    xin = xi[yin > scale]          # just look at data greater them 1 per million

    x_vec.append(xin - xin[0])



# generate plotly interactive figure

fig = go.Figure()

for j in range(n):

    fig.add_trace((go.Scatter(x=x_vec[j],y=y_vec[j],mode='lines+markers',\

                              marker=dict(size=4),

                              name=df_death_sum.index[j],\

                              text=df_death_sum.index[j])))



fig.update_xaxes(range=[0, 80])

fig.update_yaxes(range=[0, 3])

fig.update_layout(title='Countries with most Covid Deaths<br> Deaths Normalized by Million in Population',

                  #xaxis_type="log",

                  yaxis_type="log",

                  xaxis_title="Days from 1 per million in population",

                                    title_x=0.5,

                  yaxis_title="Deaths Normalized by Million in Population",

                  )
# build plot vectors

period = 7

xv = []

yv = []

for j  in range(n):

    y = df_confirmed_sum.iloc[j,:]

    yd = pd.Series(y).diff(periods=period)  # calculate weekly increases

    xv.append(y[period:len(y)])

    yv.append(yd[period:len(y)])

    

# use plotly to build interactive figure

fig = go.Figure()

for j in range(n):

    fig.add_trace((go.Scatter(x=xv[j],

                                y=yv[j],

                                marker=dict(size=4),

                                mode='lines+markers',

                                name=df_confirmed_sum.index[j],

                                text=df_confirmed_sum.index[j]))) # hover text goes here

    

fig.update_xaxes(range=[2, 6.5])

fig.update_yaxes(range=[2, 6])

fig.update_layout(title='Countries With Most Total Covid Cases',

                                    title_x=0.5,

                  xaxis_type="log",

                  yaxis_type="log",

                  xaxis_title="Total Cumulatuve Cases",

                  yaxis_title="Weekly Increase in Cases",

                  )