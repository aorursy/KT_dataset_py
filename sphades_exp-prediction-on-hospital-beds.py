# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from glob import glob

import datetime as dt

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model

from ipywidgets import interact

from IPython.core.pylabtools import figsize

from urllib.request import urlopen
from bs4 import BeautifulSoup

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
figsize(15, 9)
figsize(15, 9)

sns.set()

pd.set_option('display.max_columns', 50)

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
data_dir = "/kaggle/input/demographic-data/"
country_map = {
    'US': 'United States',
    'United States of America': 'United States',
    'Czechia': 'Czech_Republic',
    'UK': 'United_Kingdom',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'Hong Kong SAR': 'Hong Kong',
    'Russian Federation': 'Russia',
    'Mainland China': 'China',
    'Korea, South': 'South Korea',
    'Republic of Korea': 'South Korea',
}
P_SEVERE = 0.2 #WHO stats on people that require hosptialization
P_CRITICAL = 0.047
def get_line(line):
    return [x.strip() for x in line.split(',')]

def to_num(x):
    try:
        return float(x)
    except:
        return None

new_bed = os.path.join(data_dir, 'beds per 1000 data.csv')
new_rows =[]
with open(new_bed, 'r') as infile:
    headers = get_line(next(infile))
    for line in infile:
        split_liness = get_line(line)
        #print(split_liness)
        rowd = dict(zip(headers, split_liness))
        new_rows.append(rowd)
new_beds_df = pd.DataFrame(new_rows)
new_beds_df ['beds_p_1k'] = pd.to_numeric(new_beds_df['beds per 1000'], errors='coerce')
new_beds_df = new_beds_df.replace(np.nan, 0, regex=True)
new_beds_df['Country'] = new_beds_df['country']
new_beds_df = new_beds_df[['Country', 'beds_p_1k']].copy()
new_beds_df[new_beds_df['Country'].isin(['Malaysia', 'South Korea', 'Fiji'])]
pop_df = pd.read_csv(os.path.join(data_dir, 'population.csv'))
pop_df['Country'] = pop_df['Country or area']
pop_df['population'] = pop_df['Population(1 July 2019)'].map(lambda x: x.replace(',', '')).astype('int')
pop_df = pop_df[['Country', 'population']].copy()

pop_df['Country'] = pop_df['Country'].str.replace(r"\(.*\)", "")
pop_df['Country'] = pop_df['Country'].str.replace(r"\[.*\]", "")
pop_df['Country'] = pop_df['Country'].str.replace('_', ' ', regex=True)
pop_df[pop_df['Country'].isin(['Antigua and Barbuda', 'South Korea', 'Botswana'])]
new_merged_static = new_beds_df.merge(pop_df, on='Country', how='inner')
new_merged_static['beds'] = ((new_merged_static['beds_p_1k'] / 1000.0) * new_merged_static['population']).astype(int)
new_merged_static['available_beds'] = (new_merged_static['beds'] * (1/4)).astype(int) # assuming 75% occupancy rates
new_merged_static

covid_time_series_confirmed_path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'

covid_time_series_recovered_path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

covid_time_series_death_path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'


confirmed_df = pd.read_csv(covid_time_series_confirmed_path).drop(['Lat', 'Long'], axis=1)
confirmed_df = \
    confirmed_df\
        .rename(columns={'Province/State': 'State', 'Country/Region': 'Country'})\
        .melt(['State', 'Country'], var_name='Date', value_name='Confirmed')\
        .copy()

recovered_df = pd.read_csv(covid_time_series_recovered_path).drop(['Lat', 'Long'], axis=1)
recovered_df = \
    recovered_df\
        .rename(columns={'Province/State': 'State', 'Country/Region': 'Country'})\
        .melt(['State', 'Country'], var_name='Date', value_name='Recovered')\
        .copy()

deaths_df = pd.read_csv(covid_time_series_death_path).drop(['Lat', 'Long'], axis=1)
deaths_df = \
    deaths_df\
        .rename(columns={'Province/State': 'State', 'Country/Region': 'Country'})\
        .melt(['State', 'Country'], var_name='Date', value_name='Deaths')\
        .copy()

MERGE_COLS = ['State', 'Country', 'Date']
covid_df = \
    confirmed_df\
        .merge(recovered_df, on=MERGE_COLS)\
        .merge(deaths_df, on=MERGE_COLS)

covid_df = covid_df[covid_df[['Confirmed', 'Recovered', 'Deaths']].notnull().values.all(axis=1)]
covid_df['Active'] = covid_df['Confirmed'] - covid_df['Deaths'] - covid_df['Recovered']
covid_df['Severe'] = (covid_df['Active'] * P_SEVERE).astype('int64')
covid_df['Critical'] = (covid_df['Active'] * P_CRITICAL).astype('int64')
covid_df['Country'] = covid_df['Country'].map(lambda x: country_map.get(x, x.strip())) #rename some countries
covid_df['Date'] = pd.to_datetime(covid_df['Date']).dt.date
covid_df = covid_df.groupby(['Country', 'Date']).sum().reset_index().copy() #group countries, arrange by date
covid_df['Date'] = pd.to_datetime(covid_df['Date']) 

covid_df[covid_df['Country'] == 'Botswana']
countries = sorted(new_merged_static['Country'].unique())

LINEWIDTH=6

def show(country, log, forecast, beds, scale):
    
    
    sns.set(font_scale=1.5)
    country_data_df = \
        covid_df[covid_df['Country'] == country]\
            .groupby(['Country', 'Date'])\
            .sum()\
            .reset_index()

    _, row = next(country_data_df.head(1).iterrows())
    first_date = row['Date']
    
    N_DAYS = 7
    last_week = country_data_df.tail(N_DAYS)
    _, row = next(last_week.head(1).iterrows())

    first_date_last_week = row['Date']
    last_week_indices = list(range(N_DAYS))
    
    _, row = next(new_merged_static[new_merged_static['Country']==country].iterrows())
    available_beds = row['available_beds']
    
    if forecast:
        model = linear_model.LinearRegression()
        try:
            model.fit([[i] for i in last_week_indices], np.log(last_week['Severe'])) 
        except: 
            print('fail')
        #based on last 7 days of data, fit log(number of cases) into a linear regression

        if available_beds and pd.notnull(available_beds):
            try:
                N_DAYS_PREDICT = int((np.log(available_beds) - model.intercept_)/model.coef_) + 5
            except: 
                print(country)
            #predicted days = ((log(available beds) - regression intercept)/regression coef) + 5
     
                N_DAYS_PREDICT = 4 * N_DAYS #fixed 4 weeks if no predicted value

        next_week_indices = list(range(N_DAYS, N_DAYS_PREDICT))
        next_week = list(np.exp(model.predict([[i] for i in next_week_indices])))
        #using exponential model, predict values for each day until the day where it hits capacity
        next_week = [scale * n for n in next_week]
        #to adjust the scale of the graph
        
    else:
        next_week_indices = [N_DAYS]
        next_week = None
    
    predict_df = pd.DataFrame()
    predict_df['Date'] = pd.to_datetime([
        first_date_last_week + dt.timedelta(days=i) for i in next_week_indices
    ])
    predict_df['Country'] = country
    predict_df['Forecast'] = next_week
    
    concat_df = \
        country_data_df\
            .merge(predict_df, on=['Date', 'Country'], how='outer')\
            .assign(available_beds=available_beds)\
            .reset_index(drop=True)
    
    concat_df['Date'] = concat_df['Date'].dt.date
    concat_df.set_index('Date', inplace=True)
    
    ax = concat_df['Severe'].plot(logy=log, lw=LINEWIDTH, style='r-', use_index=True)
    
    positions = [p for p in concat_df.index if p.weekday() == 0]
    labels = [l.strftime('%m-%d') for l in positions]
    
    
    
    if forecast:
        concat_df['Forecast'].plot(logy=log, lw=LINEWIDTH, use_index=True, style='ro')
    
    if beds:
        concat_df['available_beds'].plot(logy=log, lw=LINEWIDTH, style='k--', xticks=[], use_index=True)
        ax.annotate(
            'Available Beds = {}'.format(int(available_beds)),
            (first_date, available_beds),
            fontsize=18,
            color='darkslategray',
            xytext=(10, -20),
            textcoords='offset points'
    )
    if forecast:
        plt.title(
            country+': Approximately {} days till hospitals exceed bed capacity'.format(N_DAYS_PREDICT - N_DAYS),
            fontsize=BIGGER_SIZE,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_facecolor("lightblue")
    plt.ylabel('Severe Cases')
    plt.show()
    #predict_df
interact(show, country=countries, log=False, forecast=True, beds=True, scale=1.0)
countries = sorted(new_merged_static['Country'].unique())

LINEWIDTH=6

global df
df = pd.DataFrame()


forecast = True;
for country in countries:
    try:

#sns.set(font_scale=1.5)
        country_data_df = \
            covid_df[covid_df['Country'] == country]\
                .groupby(['Country', 'Date'])\
                .sum()\
                .reset_index()

        _, row = next(country_data_df.head(1).iterrows())
        first_date = row['Date']

        N_DAYS = 7
        last_week = country_data_df.tail(N_DAYS)
        _, row = next(last_week.head(1).iterrows())

        first_date_last_week = row['Date']
        last_week_indices = list(range(N_DAYS))

        _, row = next(new_merged_static[new_merged_static['Country']==country].iterrows())
        available_beds = row['available_beds']

        if forecast:
            model = linear_model.LinearRegression()

            model.fit([[i] for i in last_week_indices], np.log(last_week['Severe'])) 

            #based on last 7 days of data, fit log(number of cases) into a linear regression

            if available_beds and pd.notnull(available_beds):

                N_DAYS_PREDICT = int((np.log(available_beds) - model.intercept_)/model.coef_) + 5
                #predicted days = ((log(available beds) - regression intercept)/regression coef) + 5
            else:
                N_DAYS_PREDICT = 4 * N_DAYS #fixed 4 weeks if no predicted value
                
            next_week_indices = list(range(N_DAYS, N_DAYS_PREDICT))
            next_week = list(np.exp(model.predict([[i] for i in next_week_indices])))
            #using exponential model, predict values for each day until the day where it hits capacity
            #next_week = [scale * n for n in next_week]
            #to adjust the scale of the graph
            next_week_indices = [N_DAYS]
            next_week = None

        predict_df = pd.DataFrame()
        predict_df['Date'] = pd.to_datetime([
                first_date_last_week + dt.timedelta(days=i) for i in next_week_indices
            ])
        predict_df['Country'] = country
        #predict_df['Forecast'] = next_week

        final_df = predict_df[['Country']].copy()
        final_df['Predicted Days To Shortage'] = N_DAYS_PREDICT - N_DAYS
        final_df.drop_duplicates(subset ="Country", inplace = True)
        final_df.reset_index()
        #print(final_df.to_string())
        df = df.append(final_df)
        #final_df.info(verbose=True)
        
    except: print(country)


df = df.reset_index(drop= True)
df





