from IPython.display import display

import pandas as pd

covid_data = pd.DataFrame(pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv'))

indiv_list = pd.DataFrame(pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv'))

open_list = pd.DataFrame(pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv'))

confirmed_US = pd.DataFrame(pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed_US.csv'))

confirmed = pd.DataFrame(pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv'))

deaths_US = pd.DataFrame(pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths_US.csv'))

deaths = pd.DataFrame(pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv'))

recovered = pd.DataFrame(pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv'))

 



# Check tail for most recent date

display(confirmed.tail())

# Define dates for time series

dates = confirmed.columns[4:]



#  Getting the data into a dataframe

def country_row(df,cs):

    return df[df['Country/Region']==cs]



country_row(confirmed,"Singapore")
# Worldwide count

import matplotlib.pyplot as plt 

# import matplotlib.dates as mdates

# Extract dates from one of the time series and Sum over each date value

confirmed_cases = confirmed.sum()

deaths_cases = deaths.sum()

recovered_cases = recovered.sum()



## Adjust plotting style 

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(12, 8))

# Plot the cases over extracted dates # dates = recovered.iloc[:,4:].columns

active_cases = confirmed_cases[dates]- deaths_cases[dates]-recovered_cases[dates]

plt.plot(dates, confirmed_cases[dates].values.T)

plt.plot(dates,active_cases[dates].values.T) # active cases 

plt.plot(dates, recovered_cases[dates].values.T)

plt.plot(dates, deaths_cases[dates].values.T)



# add nice label settings 

scale = 30 # scale text font and dates 

plt.xlabel('Dates', size=scale)

plt.ylabel('Number of Cases', size=scale)

plt.yticks(size=scale*2/3)

plt.title('Worldwide Count', size=20)

plt.legend(['Confirmed', 'Active', 'Recovered', 'Deaths'], prop={'size': scale*2/3}, bbox_to_anchor=(0.25, 1))



## How to better see x labels

x_ticks = range(0,dates.size, 20) # 97/8 ~= 12 slices = 13 values

plt.xticks(x_ticks,size=scale*2/3)

plt.gcf().autofmt_xdate()
# Adding that extra work into the plotting function 

def plotting_covid(dates,country,text_scale):

  import matplotlib.dates as mdates

  # Extracting the row information

  confirmed_cases = country_row(confirmed,country) 

  deaths_cases = country_row(deaths,country) 

  recovered_cases = country_row(recovered,country) 

  x = len(confirmed_cases); y = len(deaths_cases); z = len(recovered_cases)

  if x == y == z == 1:

    confirmed_cases = country_row(confirmed,country) 

    deaths_cases = country_row(deaths,country) 

    recovered_cases = country_row(recovered,country)    

  elif x == y == z > 1: 

    # Aggregate data 

    confirmed_cases = country_row(confirmed,country).sum()

    deaths_cases = country_row(deaths,country).sum()

    recovered_cases = country_row(recovered,country).sum() 

  else:

      print("Error: row sizes are not equal")

  

  # Calculate the number of active cases [Problem: cannot subtract rows easily because some row numbers are different]

  active_cases=confirmed_cases[dates].values.T-deaths_cases[dates].values.T-recovered_cases[dates].values.T

  

  # Plotting data

  plt.plot(dates, confirmed_cases[dates].values.T)

  plt.plot(dates,active_cases) # active cases 

  plt.plot(dates, recovered_cases[dates].values.T)

  plt.plot(dates, deaths_cases[dates].values.T)

  plt.xlabel('Dates', size=text_scale)

  plt.ylabel('Number of Cases', size=text_scale)

  plt.legend(['Confirmed', 'Active', 'Recovered', 'Deaths'], prop={'size': scale*2/3}, bbox_to_anchor=(0.6, 1))

  plt.yticks(size=scale*2/3)

  # How to better see x labels

  plt.xticks(size=scale*2/3)

  x_ticks = range(0,dates.size,round(text_scale)) # 97/8 ~= 12 slices = 13 values

  plt.xticks(x_ticks)

  plt.gcf().autofmt_xdate()

# Top five countries + Australia

Countries = confirmed.sort_values(by=[dates[-1]], ascending=False).head()['Country/Region']

Countries = [str(c) for c in Countries ]

Countries.append('Australia')

# Countries.append('Singapore')

# Plot Australia with top 5 countries with most confirmed cases 

# fig, ax = plt.subplots(2, 3, sharex='col', sharey='row',figsize=(15, 12))

scale = 20;

fig = plt.figure(figsize=(15,10))

fig.subplots_adjust(hspace=0.2, wspace=0.5)

for i in range(1,7): # 6 countries

    ax = fig.add_subplot(2, 3, i)

    country = Countries[i-1]

    ax.title.set_text(Countries[i-1])

    plotting_covid(dates, country, scale)
# 1. Calculate the number of active cases over time for each country

# 2. Find countries (top 10) with large descreasing trend in active cases 

# 3. Find countries (top 10) with exponential growth in COVID cases

### 1. Calculate the number of active cases over time for each country

df_active = pd.DataFrame([],columns = dates)

recorded_country = []

for country in confirmed['Country/Region'].unique():

  # Extracting the row information

  confirmed_cases = country_row(confirmed,country) 

  deaths_cases = country_row(deaths,country) 

  recovered_cases = country_row(recovered,country) 

  x = len(confirmed_cases); y = len(deaths_cases); z = len(recovered_cases)

  if x == y == z == 1:

    confirmed_cases = country_row(confirmed,country) 

    deaths_cases = country_row(deaths,country) 

    recovered_cases = country_row(recovered,country) 

    # record country 

    recorded_country.append(country)  

    # calculate active cases and put in data frame 

    data = confirmed_cases[dates].values.T-deaths_cases[dates].values.T-recovered_cases[dates].values.T

    data = pd.DataFrame(data).transpose()

    data.columns=dates

    df_active=df_active.append(data)  

  else:

    # Aggregate data 

    confirmed_cases = country_row(confirmed,country).sum()

    deaths_cases = country_row(deaths,country).sum()

    recovered_cases = country_row(recovered,country).sum() 

    # record country

    recorded_country.append(country)

    # calculate active cases and put in data frame 

    data = confirmed_cases[dates].values.T-deaths_cases[dates].values.T-recovered_cases[dates].values.T

    data = pd.DataFrame(data).transpose()

    data.columns=dates

    df_active = df_active.append(data) 



# Insert country names

df_active.insert(0, 'Country/Region',recorded_country)

df_active.head()
 # Plotting function

def plot_active(df,country_list,title,scale):

  fig, ax = plt.subplots(figsize=(14, 10))

  for i in range(len(country_list)): # 6 countries

    country = country_list[i]

    active_cases =  df[df['Country/Region']==country]

    plt.plot(dates,active_cases[dates].values.T) # active cases 

  plt.xlabel('Dates', size=text_scale)

  plt.ylabel('Number of Cases', size=text_scale)

  plt.legend(country_list, prop={'size': scale*2/3},bbox_to_anchor=(1.2, 1))

  plt.yticks(size=scale*2/3)

  plt.title(title)

  # How to better see x labels

  plt.xticks(size=scale*2/3)

  x_ticks = range(0,dates.size,round(text_scale)) # 97/8 ~= 12 slices = 13 values

  plt.xticks(x_ticks)

  plt.gcf().autofmt_xdate()

  plt.show()
display("Data Format for matplotlib",df_active.head())

dft =df_active.iloc[:,1:].transpose()

country_list = confirmed['Country/Region'].unique()

dft.columns= [ str(cl) for cl in country_list] 

dft['Dates'] = dft.index

# display("Data Format for plotly",dft.head())

df_melt = pd.melt(dft, id_vars="Dates", value_vars=dft.columns[:-1])

display("Melting data for plotly",df_melt)
df_covid = covid_data

df_covid['Active'] = df_covid['Confirmed'] - df_covid['Deaths'] - df_covid['Recovered']

df_covid = df_covid.groupby(['Country/Region','ObservationDate'],as_index=False).sum()

df_covid.tail()

# fig = px.line(df_covid[df_covid["Country/Region"]==country], x="ObservationDate", y="Active", color='Country/Region')
import plotly.express as px 

import pandas as pd 

import numpy as np 

def plotly_active(df_active,CL,title):

    dft = df_active.iloc[:,1:].transpose()

    country_list = df_active['Country/Region']

    dft.columns = [ str(cl) for cl in country_list] 

    dft['Dates'] = dft.index

    df_melt = pd.melt(dft, id_vars="Dates", value_vars = dft[CL])

    fig = px.line(df_melt, x="Dates", y="value", color='variable')

    xnum = [num for num in range(0,len(dates),20)]

    xnum.append(len(dates)-1)

    fig.update_layout(title={

        'text': title,

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

        yaxis_title="Number of Active Cases",

        xaxis = dict(

            tickmode = 'array',

            tickvals = xnum,

            ticktext = dates[xnum]

        )

    )

    fig.show()
from datetime import datetime

# check month... easier  

df4 = df_active[dates].apply(pd.to_numeric, errors='coerce')

check_dates = df4.idxmax(axis=1).values

date_num = []

for d in check_dates:

    datee = datetime.strptime(d, "%m/%d/%y")

    date_num.append(datee.month)

date_num





#FEB

df5 = pd.DataFrame({'Country/Region':country_list,'PeakMonth':date_num})

# Countries that peaked in Feb

dff = df5.loc[(df5['PeakMonth']==2) ]#| (df5['PeakMonth']==3) ,:]

# display(dff)

str(dff['Country/Region'].values)

CL = [str(el) for el in dff['Country/Region'].values ]

# plot_active(df_active,CL,"Countries with COVID Peak in Feb",20)



display("1. SELECT AREA to zoom in")

display("2. RESET AXES button")

plotly_active(df_active,CL,"Countries with COVID Peak in Feb")

len(dff)

dff.shape
# MAR 

df5 = pd.DataFrame({'Country/Region':country_list,'PeakMonth':date_num})

# Countries that peaked in Feb

dff = df5.loc[df5['PeakMonth']==3,:]

# display(dff)

str(dff['Country/Region'].values)

CL = [str(el) for el in dff['Country/Region'].values ]

display("1. HOVER over data points")

display("2. ZOOM IN button and PAN to move")

display("3. AUTOSCALE button also resets axes ")



plotly_active(df_active,CL,"Countries with COVID Peak in Mar")

#APR

df5 = pd.DataFrame({'Country/Region':country_list,'PeakMonth':date_num})

# Countries that peaked in Feb

dff = df5.loc[df5['PeakMonth']==4,:]

# display(dff)

str(dff['Country/Region'].values)

CL = [str(el) for el in dff['Country/Region'].values ]

# plot_active(df,cnames,"Countries with COVID Peak in April",20)

display("1. DOUBLE CLICK curves of interest")

display("2. Then CLICK to add other curves")

display("3. DOUBLE CLICK to reset")

plotly_active(df_active,CL,"Countries with COVID Peak in Apr")

dff.shape
#MAY 

df5 = pd.DataFrame({'Country/Region':country_list,'PeakMonth':date_num})

# Countries that peaked in Feb

dff = df5.loc[df5['PeakMonth']==5,:]

# display(dff)

str(dff['Country/Region'].values)

CL = [str(el) for el in dff['Country/Region'].values ]

# plot_active(df,cnames,"Countries with COVID Peak in May",20)

plotly_active(df_active,CL,"Countries with COVID Peak in May")
#MAY 

df5 = pd.DataFrame({'Country/Region':country_list,'PeakMonth':date_num})

# Countries that peaked in Feb

dff = df5.loc[df5['PeakMonth']==6,:]

# display(dff)

str(dff['Country/Region'].values)

CL = [str(el) for el in dff['Country/Region'].values ] 

plotly_active(df_active,CL,"Countries with COVID Peak in Jun")
#MAY 

df5 = pd.DataFrame({'Country/Region':country_list,'PeakMonth':date_num})

# Countries that peaked in Feb

dff = df5.loc[df5['PeakMonth']==7,:]

# display(dff)

str(dff['Country/Region'].values)

CL = [str(el) for el in dff['Country/Region'].values ] 

plotly_active(df_active,CL,"Countries with COVID Peak in Jul")