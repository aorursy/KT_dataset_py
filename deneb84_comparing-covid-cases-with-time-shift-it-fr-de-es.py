# import libraries

import pandas as pd



import seaborn as sns

from matplotlib import pyplot as plt

import plotly as py

import plotly.express as px



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
corona_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df = corona_df.copy()

def get_country_data(country):

    return df[df['Country/Region']==country][['ObservationDate', 'Confirmed']].groupby('ObservationDate').sum()
def fill_missing_dates(df):

    idx = pd.date_range('02-15-2020', '03-11-2020')

    df.index = pd.DatetimeIndex(df.index)

    df = df.reindex(idx, fill_value=0)

    df = df.diff(periods=1).fillna(0)

    return df



italian_cases = fill_missing_dates(get_country_data('Italy'))

french_cases = fill_missing_dates(get_country_data('France'))

german_cases = fill_missing_dates(get_country_data('Germany'))

spanish_cases = fill_missing_dates(get_country_data('Spain'))
plt.rcParams["figure.figsize"] = (20,10)

plt.plot(french_cases, label='France')

plt.plot(italian_cases, label='Italy')

plt.plot(german_cases, label='Germany')

plt.plot(spanish_cases, label='Spain')



plt.legend()
days_shift = 6



shifted_italy_cases = italian_cases.copy()

shifted_italy_cases.index = shifted_italy_cases.index.shift(days_shift, freq='D')



plt.plot(french_cases, label='France')

plt.plot(shifted_italy_cases[:-days_shift], label='Italy shift %dd' %(days_shift,))

plt.plot(german_cases, label='Germany')

plt.plot(spanish_cases, label='Spain')



plt.legend()