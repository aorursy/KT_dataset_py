import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# manipulating the default plot size

plt.rcParams['figure.figsize'] = 10,12



import warnings

warnings.filterwarnings('ignore')
# for date and time opeations

from datetime import datetime

# for file and folder operations

import os

# for regular expression opeations

import re

# for listing files in a folder

import glob

# for getting web contents

import requests 

# for scraping web contents

from bs4 import BeautifulSoup
link = 'https://www.mohfw.gov.in'

req = requests.get(link)

soup = BeautifulSoup(req.content,"html.parser")
thead = soup.find_all('thead')[-1]

head = thead.find_all('tr')

tbody = soup.find_all('tbody')[-1]

body = tbody.find_all('tr')
head_rows = []

# container for table body / contents

body_rows = []



# loop through the head and append each row to head

for tr in head:

    td = tr.find_all(['th', 'td'])

    row = [i.text for i in td]

    head_rows.append(row)

# print(head_rows)



# loop through the body and append each row to body

for tr in body:

    td = tr.find_all(['th', 'td'])

    row = [i.text for i in td]

    body_rows.append(row)
df_bs = pd.DataFrame(body_rows[:len(body_rows)-6],columns=head_rows[0])

df_bs.drop('S. No.',axis = 1, inplace=True)
df_bs.head(36)
df_India = df_bs.copy()

now = datetime.now()

df_India['Date'] = now.strftime("%m/%d/%Y")

df_India['Date'] = pd.to_datetime(df_India['Date'],format = "%m/%d/%Y")

df_India.head(36)
df_India['Name of State / UT'] = df_India['Name of State / UT'].str.replace('#', '')

df_India['Deaths**'] = df_India['Deaths**'].str.replace('#', '')



df_India = df_India.rename(columns={'Active Cases*': 'Active'})   

df_India = df_India.rename(columns={'Total Confirmed cases*': 'Confirmed'})

df_India = df_India.rename(columns={'Cured/Discharged/Migrated*':'Cured'})

df_India = df_India.rename(columns={'Name of State / UT':'State/UnionTerritory'})

df_India = df_India.rename(columns={'Name of State / UT':'State/UnionTerritory'})

df_India = df_India.rename(columns={'Deaths ( more than 70% cases due to comorbidities )':'Deaths', 

                                      'Deaths**':'Deaths'})
df_India.head(36)
df_India['Date'] = pd.to_datetime(df_India['Date'])

df_India['State/UnionTerritory'].replace('Chattisgarh', 'Chhattisgarh', inplace=True)

df_India['State/UnionTerritory'].replace('Pondicherry', 'Puducherry', inplace=True) 
# save file as a scv file

df_India.to_csv('COVID-19.csv', index=False)
df= pd.read_csv('COVID-19.csv')

df_india = df.copy()

df
df_temp = df.drop(['Date'],axis=1)

df_temp.style.background_gradient(cmap='Reds')
today = now.strftime("%Y/%m/%d")

total_activeCases = df['Active'].sum()

print("Total people who were Active as of "+today+" are: ", total_activeCases)

total_cured = df['Cured'].sum()

print("Total people who were cured as of "+today+" are: ", total_cured)

total_cases = df['Confirmed'].sum()

print("Total people who were detected COVID+ve as of "+today+" are: ", total_cases)

total_death = df['Deaths'].sum()

print("Total people who died due to COVID19 as of "+today+" are: ",total_death)
tot_active = df.groupby('State/UnionTerritory')['Active'].sum().sort_values(ascending=False).to_frame()

tot_active.style.background_gradient(cmap='Reds')
total_activeCases = df['Active'].sum()

print("Total people who were Active as of "+today+" are: ", total_activeCases)
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/06-28-2020.csv')
dates = list(confirmed_df.columns[4:])

dates = list(pd.to_datetime(dates))

dates_india = dates[8:]

df1 = confirmed_df.groupby('Country/Region').sum().reset_index()

df2 = deaths_df.groupby('Country/Region').sum().reset_index()

df3 = recovered_df.groupby('Country/Region').sum().reset_index()
k = df1[df1['Country/Region']=='India'].loc[:,'1/30/20':]

india_confirmed = k.values.tolist()[0] 



k = df2[df2['Country/Region']=='India'].loc[:,'1/30/20':]

india_deaths = k.values.tolist()[0] 



k = df3[df3['Country/Region']=='India'].loc[:,'1/30/20':]

india_recovered = k.values.tolist()[0] 
k = df1[df1['Country/Region']=='Iran'].loc[:,'1/30/20':]

iran_confirmed = k.values.tolist()[0] 



k = df2[df2['Country/Region']=='Iran'].loc[:,'1/30/20':]

iran_deaths = k.values.tolist()[0] 



k = df3[df3['Country/Region']=='Iran'].loc[:,'1/30/20':]

iran_recovered = k.values.tolist()[0] 
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 11)

plt.yticks(fontsize = 10)

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Total cases',fontsize = 20)

plt.title("Total Confirmed, Active, Death in India" , fontsize = 20)



ax1 = plt.plot_date(y= india_confirmed,x= dates_india,label = 'Confirmed',linestyle ='-',color = 'b')

ax2 = plt.plot_date(y= india_recovered,x= dates_india,label = 'Recovered',linestyle ='-',color = 'g')

ax3 = plt.plot_date(y= india_deaths,x= dates_india,label = 'Death',linestyle ='-',color = 'r')

plt.legend()
plt.figure(figsize= (15,10))

plt.xticks(rotation = 90 ,fontsize = 11)

plt.yticks(fontsize = 10)

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Total cases',fontsize = 20)

plt.title("Total Confirmed, Active, Death in Iran" , fontsize = 20)



ax1 = plt.plot_date(y= iran_confirmed,x= dates_india,label = 'Confirmed',linestyle ='-',color = 'b')

ax2 = plt.plot_date(y= iran_recovered,x= dates_india,label = 'Recovered',linestyle ='-',color = 'g')

ax3 = plt.plot_date(y= iran_deaths,x= dates_india,label = 'Death',linestyle ='-',color = 'r')

plt.legend()
from matplotlib.ticker import MaxNLocator

grouped = confirmed_df.groupby('Country/Region')

df2 = grouped.sum()

country = 'India'

MIN_CASES = 100



def make_plot(country):

    """Make the bar plot of case numbers and change in numbers line plot."""



    # Extract the Series corresponding to the case numbers for country.

    c_df = df2.loc[country, df2.columns[3:]]

    # Discard any columns with fewer than MIN_CASES.

    c_df = c_df[c_df >= MIN_CASES].astype(int)

    # Convet index to a proper datetime object

    c_df.index = pd.to_datetime(c_df.index)

    n = len(c_df)

    if n == 0:

        print('Too few data to plot: minimum number of cases is {}'

                .format(MIN_CASES))

        sys.exit(1)



    fig = plt.Figure()



    # Arrange the subplots on a grid: the top plot (case number change) is

    # one quarter the height of the bar chart (total confirmed case numbers).

    ax2 = plt.subplot2grid((4,1), (0,0))

    ax1 = plt.subplot2grid((4,1), (1,0), rowspan=3)

    ax1.bar(range(n), c_df.values)

    # Force the x-axis to be in integers (whole number of days) in case

    # Matplotlib chooses some non-integral number of days to label).

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))



    c_df_change = c_df.diff()

    ax2.plot(range(n), c_df_change.values)

    ax2.set_xticks([])



    ax1.set_xlabel('Days since {} cases'.format(MIN_CASES))

    ax1.set_ylabel('Confirmed cases, $N$')

    ax2.set_ylabel('$\Delta N$')



    # Add a title reporting the latest number of cases available.

    title = '{}\n{} cases on {}'.format(country, c_df[-1],

                c_df.index[-1].strftime('%d %B %Y'))

    plt.suptitle(title)



make_plot(country)

plt.show()
from matplotlib.ticker import MaxNLocator

grouped = confirmed_df.groupby('Country/Region')

df2 = grouped.sum()

country = 'Iran'

MIN_CASES = 100



def make_plot(country):

    """Make the bar plot of case numbers and change in numbers line plot."""



    # Extract the Series corresponding to the case numbers for country.

    c_df = df2.loc[country, df2.columns[3:]]

    # Discard any columns with fewer than MIN_CASES.

    c_df = c_df[c_df >= MIN_CASES].astype(int)

    # Convet index to a proper datetime object

    c_df.index = pd.to_datetime(c_df.index)

    n = len(c_df)

    if n == 0:

        print('Too few data to plot: minimum number of cases is {}'

                .format(MIN_CASES))

        sys.exit(1)



    fig = plt.Figure()



    # Arrange the subplots on a grid: the top plot (case number change) is

    # one quarter the height of the bar chart (total confirmed case numbers).

    ax2 = plt.subplot2grid((4,1), (0,0))

    ax1 = plt.subplot2grid((4,1), (1,0), rowspan=3)

    ax1.bar(range(n), c_df.values)

    # Force the x-axis to be in integers (whole number of days) in case

    # Matplotlib chooses some non-integral number of days to label).

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))



    c_df_change = c_df.diff()

    ax2.plot(range(n), c_df_change.values)

    ax2.set_xticks([])



    ax1.set_xlabel('Days since {} cases'.format(MIN_CASES))

    ax1.set_ylabel('Confirmed cases, $N$')

    ax2.set_ylabel('$\Delta N$')



    # Add a title reporting the latest number of cases available.

    title = '{}\n{} cases on {}'.format(country, c_df[-1],

                c_df.index[-1].strftime('%d %B %Y'))

    plt.suptitle(title)



make_plot(country)

plt.show()
!pip install Prophet
from fbprophet import Prophet

from fbprophet.plot import  plot_plotly,add_changepoints_to_plot



dates = list(confirmed_df.columns[4:])

dates = list(pd.to_datetime(dates))

dates_india = dates[8:]



k = df1[df1['Country/Region']=='India'].loc[:,'1/30/20':]

india_confirmed = k.values.tolist()[0]

k = df1[df1['Country/Region']=='Iran'].loc[:,'1/30/20':]

iran_confirmed = k.values.tolist()[0]



# for India

data_india= pd.DataFrame(columns=['ds','y'])

data_india['ds'] = dates_india

data_india['y'] = india_confirmed



# for Iran

data_iran= pd.DataFrame(columns=['ds','y'])

data_iran['ds'] = dates_india

data_iran['y'] = iran_confirmed
# Forcasting Comfirmed Cases for next 10 days for India



prop = Prophet(interval_width=0.97)

prop.fit(data_india)

future_india = prop.make_future_dataframe(periods=10)

future_india.tail(10)
forecast = prop.predict(future_india)

forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
confirmed_forecast_indiaPlot = prop.plot(forecast)
confirmed_forecast_indiaPlot =prop.plot_components(forecast)
# Forcasting Comfirmed Cases for next 10 days for Iran



prop = Prophet(interval_width=0.97)

prop.fit(data_iran)

future_iran = prop.make_future_dataframe(periods=10)

future_iran.tail(10)
forecast = prop.predict(future_iran)

forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
confirmed_forecast_iranPlot = prop.plot(forecast)
confirmed_forecast_iranPlot =prop.plot_components(forecast)