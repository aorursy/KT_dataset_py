# imports

import os

from operator import itemgetter

import numpy as np

import pandas as pd

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt



# constants

pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 200)

colors = {

  'very_light_gray': '#ececec',

  'light_gray': '#b6b6b6',

  'medium_gray': '#929292',

  'very_dark_gray': '#414141',

  'orange': '#ff6f00',

  'light_blue': '#79c3ff',

  'light_purple': '#d88aff',

  'light_green': '#b4ec70',

  'light_yellow': '#fff27e',

  'light_red': '#ff7482',

  'light_cyan': '#84ffff'

}

start_date = np.datetime64('2020-01-22')

all_dates = [start_date + np.timedelta64(x, 'D') for x in range(0, 100)]
# converts a country's data into a time series dataframe

def convert_to_ts (data, country):

  df = data[data['Country/Region'] == country].groupby(['Date'], as_index=False)['ConfirmedCases'].sum()

  df.columns = ['date', 'count']

  df['date'] = df['date'].astype('datetime64[ns]')

  return df



data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

dat = [

  { 'name': 'China', 'color': 'light_gray' },

  { 'name': 'Korea, South', 'color': 'medium_gray' },

  { 'name': 'Italy', 'color': 'very_dark_gray' },

  { 'name': 'Iran', 'color': 'light_blue' },

  { 'name': 'Spain', 'color': 'light_purple' },

  { 'name': 'Germany', 'color': 'light_green' },

  { 'name': 'France', 'color': 'light_yellow' },

  { 'name': 'United Kingdom', 'color': 'light_red' },

  { 'name': 'Switzerland', 'color': 'light_cyan' },

  { 'name': 'US', 'color': 'orange' },

]

countries = { d['name']: convert_to_ts(data, d['name']) for d in dat}
pd.DataFrame(

  map(lambda obj: [obj[0], f'{obj[1]["count"].max():,.0f}'], countries.items()),

  columns=['Country', 'Max infected']

)
def comparison_to_china_penalty (df, offset):

  china_counts = countries['China']['count'].to_numpy()

  counts = df['count'].to_numpy()

  residuals = []

  for i in range(0, 7):

    if i + offset < len(counts):

      residuals.append(china_counts[i] - counts[i + offset])

    else:

      residuals.append(0)

  return np.power(residuals, 2).sum()



def find_optimal_offset (df):

  penalties = []

  for offset in range(len(df)):

    penalties.append(comparison_to_china_penalty(df, offset))

  return np.argmin(penalties)



for d in dat:

  d['offset'] = find_optimal_offset(countries[d['name']])

dat.sort(key=lambda x: x['offset'])
pd.DataFrame(dat, columns=['name', 'offset'])
fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(111)



for d in dat:

  country_name, offset, color_key = itemgetter('name', 'offset', 'color')(d)

  country = countries[country_name]

  ax.plot(

    country['date'] - np.timedelta64(offset, 'D'),

    country['count'],

    label=f'{country_name} = China - {offset} days',

    color=colors[color_key]

  )



plt.xlim((np.datetime64('2020-01-22'), np.datetime64('2020-02-22')))

plt.xticks([np.datetime64('2020-01-22') + np.timedelta64(d, 'D') for d in range(0, 15)])

ax.set_xticklabels(range(0, 15))

plt.xlabel('Days since onset for each Country')



plt.ylim((0, 30000))

ax.set_yticklabels(['0' if x == 0 else '{:.0f}k'.format(int(x) / 1000) for x in ax.get_yticks().tolist()])

plt.ylabel('Confirmed infections')



plt.legend(title='Countries', loc='lower right')



plt.show()

def sigmoid (x, A, slope, offset):

  return A / (1 + np.exp ((x - (offset + 17.75)) / slope))



def fit_to_sigmoid (df, offset, all_dates):

  dates = (df['date'] - start_date) / np.timedelta64(1, 'D')

  p, _ = curve_fit(

    lambda x, A, slope: sigmoid(x, A, slope, offset),

    dates,

    df['count'],

    p0=[80000, -5],

    bounds=(

      [-np.inf, -np.inf],

      [np.inf, -0.01]

    ),

    maxfev=5000,

  )

  return sigmoid((all_dates - start_date) / np.timedelta64(1, 'D'), *p, offset), p



for d in dat:

  country_name, offset, color_key = itemgetter('name', 'offset', 'color')(d)

  country = countries[country_name]

  fit, p = fit_to_sigmoid(country, offset, all_dates)

  d['fit'] = fit

  d['p'] = p
china_slope = dat[0]['p'][1]

growth_rate_relative_to_china = lambda p: china_slope/p[1]



table_data = []

for d in dat:

  country_name, offset, p = itemgetter('name', 'offset', 'p')(d)

  # name, days behind china, relative growth rate, max infected

  table_data.append([

    country_name, 

    '' if country_name == 'China' else f'{offset}',

    f'{growth_rate_relative_to_china(p):.1f}',

    f'{p[0]:,.0f}'

  ])



pd.DataFrame(table_data, columns=['Country', 'Days behind China', 'Growth rate relative to China', 'Estimated max infected'])
fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(111)



for d in dat:

  country_name, color_key, fit = itemgetter('name', 'color', 'fit')(d)

  country = countries[country_name]

  ax.plot(

    country['date'],

    country['count'],

    label=country_name,

    color=colors[color_key],

    linewidth=3

  )

  ax.plot(

    all_dates,

    fit,

    color=colors[color_key],

    linestyle=':'

  )



# plots the "now" line

y_max = 150000

now = np.datetime64('2020-03-18').astype('datetime64[D]')

plt.vlines(now, ymin=0, ymax=y_max, colors=colors['very_light_gray'], linestyles='dashed')

plt.annotate('Actual', xy=(now - np.timedelta64(1, 'D'), y_max - 5000), ha='right', va='top')

plt.annotate('Estimated', xy=(now + np.timedelta64(1, 'D'), y_max - 5000), ha='left', va='top')



ticks = [np.datetime64('2020-02-01') + np.timedelta64(7 * x, 'D') for x in range(0, 15)]

label_from_tick = lambda tick: pd.to_datetime(tick).strftime('%b %d')

tick_labels = list(map(label_from_tick, ticks))

plt.xticks(ticks, tick_labels, rotation=20, horizontalalignment='right')



plt.ylim((0, y_max))

ax.set_yticklabels(['0' if x == 0 else '{:.0f}k'.format(int(x) / 1000) for x in ax.get_yticks().tolist()])

plt.ylabel('Confirmed infections')



plt.grid(color=colors['very_light_gray'])



plt.legend(title='Countries', loc='lower right')



plt.show()