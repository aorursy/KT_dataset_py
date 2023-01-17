import requests, json

import pandas as pd

import numpy as np

from bs4 import BeautifulSoup as bs

import matplotlib.pyplot as plt

import seaborn as sns

import re

import datetime as dt

import os
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
df.head()
df.info()
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d') #adjusting format
df['Country/Region'] = df['Country/Region'].str.replace('*','') #adjusting Taiwan's name
#checking which countries have state

df['Country/Region'][df['Province/State'].notnull()].unique()
df['Province/State'][df['Country/Region'] == 'Denmark'].unique()
for country in df['Country/Region'][df['Province/State'].notnull()].unique():

    print ('Country:',country)

    states = df['Province/State'][df['Country/Region'] == country].unique()

    print('States:', states)
country = ['Denmark', 'France','Netherlands','United Kingdom']

df['Region'] = np.where(df['Country/Region'].isin(country),df['Province/State'],df['Country/Region'])

#df['Region'] = np.select(conditions, choices)
#checking those countries:

for country in df['Country/Region'][df['Province/State'].notnull()].unique():

    print ('Country:', country)

    region = df['Region'][df['Country/Region'] == country].unique()

    print('Region:', region)
df2 = df.groupby(['Date','Region'])[["ConfirmedCases", "Fatalities"]].sum().reset_index() #id, latitude and longitude
df2.sort_values(['Region','Date'], inplace = True)
df2[df2['Region'] == 'US']
def first_case(df,place_id,col,place):

    min_cases = df[col][df[col] != 0][df[place] == place_id]

    if len(min_cases) == 0:

        min_cases = 0

        date = df['Date'][df[place] == place_id].max()

        #print('0:', place_id, date)

    else:

        min_cases = df[col][df[col] != 0][df[place] == place_id].min()

        date = df['Date'][df[col] == min_cases][df[place] == place_id].min()

        #print(min_cases, place_id, date)

    return date
first_cases = {}

deaths = {}

place = 'Region' # 'Province/State' or 'Country/Region'

for country in df[place].unique():

    date = first_case(df2, country,'ConfirmedCases',place)

    date_death = first_case(df2, country,'Fatalities',place)

    first_cases[country] = date #where there's no case/death yet, the lastest date is considered

    deaths[country] = date_death
#Correcting the dates of the first case and first death of China
#first_cases['China'] = 2019-12-??
df2['1st_case'] = df2[place].map(first_cases)

df2['1st_fatality'] = df2[place].map(deaths)
df2['Days_since_1st_case'] = (df2['Date'] - df2['1st_case']).dt.days

df2['Days_since_1st_fatality'] = (df2['Date'] - df2['1st_fatality']).dt.days
sns.lineplot(x='Days_since_1st_case', y="ConfirmedCases",

             hue="Region", legend = False,

             data = df2[df2['Days_since_1st_case'] >= 0])



plt.title('Cases vs days since the first case')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xticks(rotation=0)



#As we can see, in some countries the spread took longer to break
sns.lineplot(x='Days_since_1st_case', y="Fatalities",

             hue="Region", legend = False,

             data = df2[df2['Days_since_1st_case'] >= 0])

plt.title('Fatalities vs days since the first case')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xticks(rotation=0)
def daily_var(df, column_name, common_value,shift_value):

    df[column_name] = np.where(df[common_value] == df[common_value].shift(1), df[shift_value].diff(1),0)
daily_var(df2,'new_cases','Region','ConfirmedCases')
daily_var(df2,'new_fatalities','Region','Fatalities')
plt.figure(figsize=(15, 5))

sns.barplot(x = 'Days_since_1st_case', y = 'new_cases', data = df2[df2['Region'] == 'Italy'])

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xticks(rotation=0)
def daily_evolution(df,region):

    labels = df['Days_since_1st_case'][df['Region'] == region].unique()

    cases = df['new_cases'][df['Region'] == region]

    #fatalities = df['new_fatalities'][df['Region'] == region]

    width = 0.35       # the width of the bars: can also be len(x) sequence



    fig, ax = plt.subplots()



    ax.bar(labels, cases, width, label='Cases')

    #ax.bar(labels, fatalities, width, bottom=cases,label='Fatalities')



    ax.set_ylabel('Total')

    ax.set_xlabel('Days since 1st case')

    ax.set_title('New cases by day in ' + region)

    ax.legend()



    plt.show()
daily_evolution(df2,'Spain')
def growth_factor(df, column_name, place,shift_value):

    df[column_name] = np.where(df[place] == df[place].shift(1), df[shift_value].div(df[shift_value].shift(1)),0)

    df[column_name].replace([np.inf,-np.inf],np.nan, inplace = True)

    df[column_name].fillna(method = 'ffill', inplace = True)
#growth factor moving average

def GFMA(df, column_name, place,shift_value, wndw):

    df[column_name] = np.where(df[place] == df[place].shift(1), df[shift_value].rolling(window = wndw).mean(),0)

    df[column_name].fillna(method = 'ffill', inplace = True)
growth_factor(df2,'growth_factor','Region','new_cases')

GFMA(df2,'GFMA','Region','growth_factor',14) #14 days because theoretically that's how long it takes for the virus to be killed in our organism

growth_factor(df2,'growth_factor_f','Region','new_fatalities')

GFMA(df2,'GFMA_f','Region','growth_factor_f',14)
df2.head()
plt.figure(figsize=(15, 5))

sns.barplot(x = 'Days_since_1st_case', y = 'GFMA', hue = 'Region', data = df2[df2['Region'] == 'China'])

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xticks(rotation=0)
def bar_n_line_graph(df,x_axis,bar_values, line_values,region):

      

    temp = df[df['Region'] == region]

    

    fig, ax = plt.subplots()

    labels = temp[x_axis]

    width = 0.35       # the width of the bars: can also be len(x) sequence

    

    ax2 = ax.twinx()

    

    bar = temp[bar_values]

    line = temp[line_values]

    

    ax.plot(kind = 'bar', x = labels, height = bar, label='Growth Factor')

    ax2.plot(labels, line, width, ax = ax2,label='GFMA')



    #ax.set_ylabel('Total')

    #ax.set_title('New cases by day in ' + region)

    #ax.legend()

    

    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    #plt.xticks(rotation=45)

    #plt.figure(figsize=(15, 10))

    plt.show()
df2[df2['Region'] == 'China'].plot(x = 'Days_since_1st_case', y = ['GFMA','GFMA_f'])
avg_temp_url = 'https://en.wikipedia.org/wiki/List_of_cities_by_average_temperature'

wiki_text = requests.get(avg_temp_url).text

soup = bs(wiki_text, 'html.parser')

tables = soup.find_all('table')

table_str = str(tables)
continents = []

for i in range (0,6):

    t = pd.read_html(table_str)[i]

    continents.append(t)

temps = pd.concat(continents)

temps.drop('Ref.', axis = 1, inplace = True)
#adjusting names

temps['Country'].replace(['United States'],'US', inplace = True)

temps['Country'].replace(['United ArabEmirates'],'United Arab Emirates', inplace = True)

temps['Country'].replace(['South Korea'],'Korea, South', inplace = True)

temps['Country'].replace(['The '],'', inplace = True)
for month in temps.iloc[:,2:].columns:

    temps[month] = temps[month].str.replace('−','-').str.split('(', expand = True)[0]

    #temps[month] = temps[month].apply(pd.to_numeric, errors='ignore')
temps = temps.apply(pd.to_numeric, errors = 'ignore')
avg_temps = temps.groupby('Country').mean() #monthly average for each country
plt.figure(figsize=(30, 10))



sns.barplot(x = avg_temps.index, y = 'Year', data = avg_temps.sort_values('Year', ascending = False))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.xticks(rotation=90)
def replace_names_who(df):

    df['Country Name'].replace(['United States'],'US', inplace = True)

    #temps['Country'].replace(['United ArabEmirates'],'United Arab Emirates', inplace = True)

    #temps['Country'].replace(['South Korea'],'Korea, South', inplace = True)

    #temps['Country'].replace(['The '],'', inplace = True)
latest = df2['Date'].max()

latest
df_latest = df2[df2['Date'] == latest].copy()
df_latest.sort_values('growth_factor', inplace = True, ascending = False)
df_latest.head()
files = os.listdir('/kaggle/input/world-health-organisation')
for i in range(1, len(files)):

    file = '/kaggle/input/world-health-organisation/' + files[i]

    name = files[i][:-4]

    temp = pd.read_csv(file, skiprows = 3)

    replace_names_who(temp)

    temp.head()

    

    #finds the column with the latest data

    row = 2

    while temp[temp.columns[-row]].notnull().mean() == 0:

        col = temp.columns[-row]

        #print(name,'row:',row, 'mean:' )

        #print(temp[temp.columns[-row]].notnull().mean())

        row += 1

        #print('new row:',row)

        

    temp.set_index('Country Name', inplace = True)

    col = temp.columns[-row]

    #print('col:',col)

    #print(temp[col])

    

    #consolidating dataset

    df_latest[name] = df_latest['Region'].map(temp[col])

    df_latest[name].fillna(temp.loc['World',col], inplace = True) #fills empty values with World mean

    

    df2[name] = df2['Region'].map(temp[col])

    df2[name].fillna(temp.loc['World', col], inplace = True)
df_latest['Avg_temp'] = df_latest['Region'].map(avg_temps[['Jan','Feb']].mean(axis = 1))

df_latest['Avg_temp'] = df_latest['Avg_temp'].fillna(df_latest['Avg_temp'].mean())#fills empty values with global mean



df2['Avg_temp'] = df2['Region'].map(avg_temps[['Jan','Feb']].mean(axis = 1))

df2['Avg_temp'] = df2['Avg_temp'].fillna(df2['Avg_temp'].mean())#fills empty values with global mean
df_latest.head()
df2.head()
sns.pairplot(df_latest)
raw = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
raw.head()
def preprocess_test(df):

    df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d') #adjusting format

    df['Country/Region'] = df['Country/Region'].str.replace('*','') #adjusting Taiwan's name

    for country in df['Country/Region'][df['Province/State'].notnull()].unique():

        #print ('Country:',country)

        states = df['Province/State'][df['Country/Region'] == country].unique()

        #print('States:', states)

    country = ['Denmark', 'France','Netherlands','United Kingdom']

    df['Region'] = np.where(df['Country/Region'].isin(country),df['Province/State'],df['Country/Region'])

    #df['Region'] = np.select(conditions, choices)

    df2 = df.groupby(['Date','Region'])[["Lat", "Long"]].mean().reset_index() #id, latitude and longitude

    df2.sort_values(['Region','Date'], inplace = True)

    df2.drop(['Lat','Long'],axis = 1, inplace = True)

    

    df2['1st_case'] = df2[place].map(first_cases)

    df2['1st_fatality'] = df2[place].map(deaths)

    df2['Days_since_1st_case'] = (df2['Date'] - df2['1st_case']).dt.days

    df2['Days_since_1st_fatality'] = (df2['Date'] - df2['1st_fatality']).dt.days

    

    return df2
test = preprocess_test(raw)
from statsmodels.graphics import tsaplots

import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.arima_process import arma_generate_sample

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing, Holt
def forecast(df_train, df_test,col,country,period,training_size = 0.8):

    

    temp = df_train[df_train['Region'] == country].copy()

    temp.set_index(pd.DatetimeIndex(temp['Date']), inplace = True)

    #temp.drop('Date', axis = 1, inplace = True)

    

    temp_pred = df_test[df_test['Region'] == country].copy()

    

    total_days = temp['Days_since_1st_case'].max()

    

    #Time Series Properties

    fig, ax = plt.subplots(1,2,figsize=(15,5))

    tsaplots.plot_acf(temp[col], lags= 24, ax=ax[0])

    tsaplots.plot_pacf(temp[col], lags= 24, ax=ax[1])

    plt.show()

    

    decomposition = sm.tsa.seasonal_decompose(temp[col], period = 12)

    fig = decomposition.plot()

    plt.show()

   

    #Model

    train = temp[col][temp['Days_since_1st_case'] <= total_days*training_size] #training default cut to 80% of dataset

    test = temp[col][temp['Days_since_1st_case'] > total_days*training_size]

    

    

    es = SimpleExpSmoothing(train).fit(smoothing_level=0.5)

    fore_es = es.forecast(temp_pred.shape[0])

     

    #holt = Holt(df.qtde[:'2018']).fit()

    #fore = holt.forecast(df.qtde['2018':].shape[0])

    holt = Holt(train).fit()

    fore_holt = holt.forecast(temp_pred.shape[0])

    pred_holt = holt.predict(temp_pred.shape[0])

        

    hw = ExponentialSmoothing(train, seasonal_periods=period, trend='add',seasonal='add').fit()

    fore_hw = hw.forecast(temp_pred.shape[0])

    #pred_hw = holt.predict(temp_pred.shape[0])

    

    plt.figure(figsize=(16,8))

    plt.plot(temp[col], label='Train')

    plt.plot(test, label='Test')

    plt.plot(fore_hw, label='Holt_Winters')

    plt.plot(fore_es, label='Exponential Smoothing')

    plt.plot(fore_holt, label='Holt Linear')

    

    #plt.xticks(rotation=45)

    plt.legend(loc='best')

    plt.show()
forecast(df2,test,'ConfirmedCases','Italy', 25, 1)