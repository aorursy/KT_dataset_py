import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'])
df_train
df_train['FirstDateCountry'] = df_train.query('ConfirmedCases>0').groupby('Country/Region')['Date'].transform('min')

df_train['First10DateCountry'] = df_train.query('ConfirmedCases>=10').groupby('Country/Region')['Date'].transform('min')

df_train['First50DateCountry'] = df_train.query('ConfirmedCases>=50').groupby('Country/Region')['Date'].transform('min')
df_train['DaysSinceFirstInCountry'] = (df_train['Date'] - df_train['FirstDateCountry']).dt.days

df_train['DaysSinceFirst10InCountry'] = (df_train['Date'] - df_train['First10DateCountry']).dt.days

df_train['DaysSinceFirst50InCountry'] = (df_train['Date'] - df_train['First50DateCountry']).dt.days
fig, ax = plt.subplots(figsize=(20, 10))

ax.set_title('DaysSinceFirstInCountry')

for country, df_country in df_train.query('DaysSinceFirstInCountry > 0').groupby('Country/Region'):

    df_country_plot = df_country.set_index('DaysSinceFirstInCountry').sort_index()['ConfirmedCases'].cummax()

    df_country_plot.plot(label=country, ax=ax)

    ax.annotate(country, (df_country_plot.index[-1], df_country_plot.iloc[-1]))

ax.set_yscale('log')
fig, ax = plt.subplots(figsize=(20, 10))

ax.set_title('DaysSinceFirst10InCountry')

for country, df_country in df_train.query('DaysSinceFirst10InCountry > 0').groupby('Country/Region'):

    df_country_plot = df_country.set_index('DaysSinceFirst10InCountry').sort_index()['ConfirmedCases'].cummax()

    df_country_plot.plot(label=country, ax=ax)

    ax.annotate(country, (df_country_plot.index[-1], df_country_plot.iloc[-1]))

ax.set_yscale('log')
fig, ax = plt.subplots(figsize=(20, 10))

ax.set_title('DaysSinceFirst50InCountry')

for country, df_country in df_train.query('DaysSinceFirst50InCountry > 0').groupby('Country/Region'):

    df_country_plot = df_country.set_index('DaysSinceFirst50InCountry').sort_index()['ConfirmedCases'].cummax()

    df_country_plot.plot(label=country, ax=ax)

    ax.annotate(country, (df_country_plot.index[-1], df_country_plot.iloc[-1]))

ax.set_yscale('log')