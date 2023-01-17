import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime



# hide warnings

import warnings

warnings.filterwarnings('ignore')
#check the old format

confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

death_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')



confirmed_table = confirmed_df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Confirmed").fillna('').drop(['Lat', 'Long'], axis=1)

death_table = death_df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Deaths").fillna('').drop(['Lat', 'Long'], axis=1)

recovered_table = recovered_df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Recovered").fillna('').drop(['Lat', 'Long'], axis=1)



full_table = confirmed_table.merge(death_table).merge(recovered_table)



full_table['Date'] = pd.to_datetime(full_table['Date'])

full_table
# cases 

#cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']



# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')

# full_table[cases] = full_table[cases].fillna(0)

full_table
def get_time_series(country):

    # for some countries, data is spread over several Provinces

    if full_table[full_table['Country/Region'] == country]['Province/State'].nunique() > 1:

        country_table = full_table[full_table['Country/Region'] == country]

        country_df = pd.DataFrame(pd.pivot_table(country_table, values = ['Confirmed', 'Deaths', 'Recovered', 'Active'],

                              index='Date', aggfunc=sum).to_records())

        return country_df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']]

    df = full_table[(full_table['Country/Region'] == country) 

                & (full_table['Province/State'].isin(['', country]))]

    return df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']]





def get_time_series_province(province):

    # for some countries, data is spread over several Provinces

    df = full_table[(full_table['Province/State'] == province)]

    return df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']]
country = 'Indonesia'

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

print('Indonesia COVID-19 Prediction')

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



df = full_table[full_table['Country/Region'] != 'China'][['Province/State','Country/Region', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'Active']].groupby('Date').sum()

print('World COVID-19 Prediction')

opt_display_model(df, stats)
# Country Specify

stats = []

for country in ['US', 'United Kingdom', 'Russia', 'Singapore', 'New Zealand']:

# for country in ['Sweden']:

    df = get_time_series(country)



    print('{} COVID-19 Prediction'.format(country))

    

    opt_display_model(df, stats)
# All Country Confirmed Greater than 1000



stats = []

for country in sorted(full_table['Country/Region'].unique()):

    df = get_time_series(country)

    # only consider countries with at least 1000 cases (plus Sweden)

    if len(df) == 0 or (max(df['Confirmed']) < 1000): 

        continue

    print('{} COVID-19 Prediction'.format(country))

    opt_display_model(df, stats)


stats_df = pd.DataFrame(stats)

# stats_df.columns = ['country', 'N', 'a', 'alpha', 'lag']

stats_df.columns = ['country', 'Confirmed-N', 'Confirmed-a', 'Confirmed-alpha', 'Deaths-N', 'Deaths-a', 'Deaths-alpha', 'Recorved-N', 'Recorved-a', 'Recorved-alpha']

stats_df
pd.set_option('display.max_rows', 500)

pd.options.display.float_format = '{:20,.4f}'.format

stats_df.astype({'Confirmed-N': 'int'}).sort_values(by='Confirmed-N', ascending=False)
ax = stats_df.plot.scatter(x='Confirmed-alpha', y='Confirmed-a')

# ax.set_xlim([0, 100])

plt.show()
stats = []

for Province in sorted(full_table['Province/State'].unique()):

    if (Province == ''):

        continue

    df = get_time_series_province(Province)

    # only consider Province with at least 500 cases

    if len(df) == 0 or (max(df['Confirmed']) < 500): 

        continue

    print('{} COVID-19 Prediction'.format(Province))

    opt_display_model(df, stats)
#check the old format

fulltable_us = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')

fulltable_us = fulltable_us.drop(['fips'], axis=1).groupby(['date','state']).sum().reset_index()

fulltable_us.columns = ['Date', 'Province/State', 'Confirmed', 'Deaths']

fulltable_us['Date'] = pd.to_datetime(fulltable_us['Date'])



fulltable_us
def display_fit_us(df, opt_confirmed, opt_deaths, ax):

    model_x = []

    

    isValid = True

    last_death_rate = 0

    

    for t in range(len(df)):

        model_x.append([df.index[t], model(*opt_confirmed, t), model(*opt_deaths, t)])

        

        #if deaths + recovered > confirmed or deaths rate > 5%, maybe not valid

        if (t > len(df)):

            last_row = model_x[-1]

            if (last_row[2] > last_row[1]*0.05):

                if (isValid):

                    last_row2 = model_x[-2]

                    last_death_rate = last_row2[2]/last_row2[1]

                    isValid = False

                    

            if (last_row[2] > last_row[1]*0.05):

                last_row[2] = last_row[1]*last_death_rate

                

                

    model_sim = pd.DataFrame(model_x, dtype=int)

    model_sim.set_index(0, inplace=True)

    model_sim.columns = ['Model-Confirmed', 'Model-Deaths']

    plot_color = ['#99990077', '#FF000055', '#999900FF', '#FF0000FF']



    return pd.concat([model_sim, df], axis=1).plot(ax=ax, figsize=(14, 10), color = plot_color)



def display_extended_curve_us(df, opt_confirmed, opt_deaths, ax):

    start_date = df.index[0]

    n_days = len(df) + 40

    extended_model_x = []

    

    isValid = True

    last_death_rate = 0

    

    for t in range(n_days):

        extended_model_x.append([start_date + datetime.timedelta(days=t), model(*opt_confirmed, t), model(*opt_deaths, t)])

        

        #if deaths + recovered > confirmed or deaths rate > 5%, maybe not valid

        if (t > len(df)):

            last_row = extended_model_x[-1]

            if (last_row[2] > last_row[1]*0.05):

                if (isValid):

                    last_row2 = extended_model_x[-2]

                    last_death_rate = last_row2[2]/last_row2[1]

                    isValid = False

            

            if (last_row[2] > last_row[1]*0.05):

                last_row[2] = last_row[1]*last_death_rate

                    

                

                

    extended_model_sim = pd.DataFrame(extended_model_x, dtype=int)

    extended_model_sim.set_index(0, inplace=True)

    extended_model_sim.columns = ['Model-Confirmed', 'Model-Deaths']



    plot_color = ['#99990077', '#FF000055', '#999900FF', '#FF0000FF']



    return pd.concat([extended_model_sim, df], axis=1).plot(ax=ax, figsize=(14, 10), color = plot_color)





def opt_display_model_us(df, stats):

    # if the last data point repeats the previous one, or is lower, drop it

    if len(df) > 1 and df.iloc[-2,0] >= df.iloc[-1,0]:

        df.drop(df.tail(1).index,inplace=True)

    global model_index

    model_index = 0

    opt_confirmed = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

    model_index = 1

    opt_deaths = minimize(model_loss, x0=np.array([200000, 0.05, 15]), method='Nelder-Mead', tol=1e-5).x

    if min(opt_confirmed) > 0:

        stats.append([country, *opt_confirmed, *opt_deaths])

        n_plot = len(stats)

        plt.figure(1)

        ax1 = plt.subplot(221)

        display_fit_us(df, opt_confirmed, opt_deaths, ax1)

        ax2 = plt.subplot(222)

        display_extended_curve_us(df, opt_confirmed, opt_deaths, ax2)

        plt.show()





def get_time_series_province_us(province):

    # for some countries, data is spread over several Provinces

    global fulltable_us

    df = fulltable_us[(fulltable_us['Province/State'] == province)]

    return df.set_index('Date')[['Confirmed', 'Deaths']]
df = get_time_series_province_us('Washington')

df
stats = []

print('{} of United States COVID-19 Prediction'.format('Washington'))

opt_display_model_us(df, stats)
stats = []

for Province in sorted(fulltable_us['Province/State'].unique()):

    if (Province == ''):

        continue

    df = get_time_series_province_us(Province)

    if len(df) == 0 or (max(df['Confirmed']) < 500): 

        continue

    print('{} of United States COVID-19 Prediction'.format(Province))

    opt_display_model_us(df, stats)