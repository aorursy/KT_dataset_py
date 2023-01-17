# imports

import os

import sys

from operator import itemgetter

import numpy as np

import pandas as pd

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
# constants

pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 200)

pd.plotting.register_matplotlib_converters()

n_jobs = -1

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
#%% load data

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv', parse_dates=['Date'])

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv', parse_dates=['Date'])



print('Original training data')

train.head()



start_date, cutoff_date, max_date = train['Date'].to_numpy()[0], np.datetime64('2020-03-24'), test['Date'].to_numpy()[-1]

data = train[train['Date'] <= cutoff_date].copy()

validation_data = train[train['Date'] >= cutoff_date].copy()

validation_data['ConfirmedCasesActual'] = validation_data['ConfirmedCases']

validation_data['ConfirmedCases'] = np.NaN

validation_data['FatalitiesActual'] = validation_data['Fatalities']

validation_data['Fatalities'] = np.NaN

dates, validation_dates, test_dates = data['Date'].unique(), validation_data['Date'].unique(), test['Date'].unique()

all_dates = np.unique(np.concatenate((dates, test_dates)))



data['ForecastId'] = -1

validation_data['ForecastId'] = -1

test_data = test.copy()

test_data['ConfirmedCases'] = np.NaN

test_data['Fatalities'] = np.NaN
#%% create unique area key

def unique_area_key (row):

  if pd.isnull(row['Province/State']): return row['Country/Region']

  return f'{row["Country/Region"]} {row["Province/State"]}'

data['unique_area_key'] = data.apply(unique_area_key, axis=1)

validation_data['unique_area_key'] = validation_data.apply(unique_area_key, axis=1)

test_data['unique_area_key'] = test_data.apply(unique_area_key, axis=1)
#%% function fitting definitions

def get_counts_for_key (df, key):

  return df[df['unique_area_key'] == key][['Date', 'ConfirmedCases', 'Fatalities']]



def rmsle(y, y0):

  return np.sqrt(np.mean(np.square(np.log1p(y[-15:]) - np.log1p(y0[-15:]))))



def sigmoid (x, A, slope, offset):

  return A / (1 + np.exp ((x - (offset + 17.75)) / slope))



def fit_to_sigmoid (dates, actual_counts, all_dates):

  max_counts = actual_counts.max() + 1

  x = (dates - start_date) / np.timedelta64(1, 'D')

  p, _ = curve_fit(

    sigmoid,

    x,

    actual_counts,

    p0=[max_counts / 2.0, -5, 10],

    bounds=(

      [0, -np.inf, 0],

      [max_counts * 10.0, -2.0, 100]

    ),

    maxfev=5000,

  )

  return (

    sigmoid((all_dates - start_date) / np.timedelta64(1, 'D'), *p),

    p,

    rmsle(sigmoid((dates - start_date) / np.timedelta64(1, 'D'), *p), actual_counts)

  )



def exponential (x, A, B, cutoff):

  y_1 = A * x[x < cutoff]

  y_2 = B * (x[x >= cutoff] - cutoff) + A * x[x >= cutoff][0]

  return np.concatenate((y_1, y_2))



def fit_to_exponential (dates, actual_counts, all_dates):

  x = (dates - start_date) / np.timedelta64(1, 'D')

  p, _ = curve_fit(

    exponential,

    x,

    actual_counts,

    p0=[0.01, 1, x[1]],

    bounds=(

      [0, 0, x[0]],

      [np.inf, np.inf, x[-1]]

    ),

    maxfev=10000,

  )

  return (

    exponential((all_dates - start_date) / np.timedelta64(1, 'D'), *p),

    p,

    rmsle(exponential((dates - start_date) / np.timedelta64(1, 'D'), *p), actual_counts)

  )
#%% compute optimal parameters for each country

keys = data['unique_area_key'].unique()

params = {}

for index, key in enumerate(keys):

  print(f'Running key {index + 1} of {len(keys)}: {key}')

  counts = get_counts_for_key(data, key)

  # sigmoid

  try:

    confirmed_cases_sigmoid, confirmed_cases_sigmoid_p, confirmed_cases_sigmoid_score = fit_to_sigmoid(counts['Date'], counts['ConfirmedCases'], all_dates)

  except:

    print(f'Failed to fit sigmoid on confirmed cases: {sys.exc_info()}')

    confirmed_cases_sigmoid, confirmed_cases_sigmoid_p, confirmed_cases_sigmoid_score = None, None, None

  

  try:

    fatalities_sigmoid, fatalities_sigmoid_p, fatalities_sigmoid_score = fit_to_sigmoid(counts['Date'], counts['Fatalities'], all_dates)

  except:

    print(f'Failed to fit sigmoid on fatalities: {sys.exc_info()}')

    fatalities_sigmoid, fatalities_sigmoid_p, fatalities_sigmoid_score = None, None, None

  

  # exponential

  try:

    confirmed_cases_exp, confirmed_cases_exp_p, confirmed_cases_exp_score = fit_to_exponential(counts['Date'].to_numpy(), counts['ConfirmedCases'], all_dates)

  except:

    print(f'Failed to fit exp on confirmed cases: {sys.exc_info()}')

    confirmed_cases_exp, confirmed_cases_exp_p, confirmed_cases_exp_score = None, None, None

  try:

    fatalities_exp, fatalities_exp_p, fatalities_exp_score = fit_to_exponential(counts['Date'].to_numpy(), counts['Fatalities'], all_dates)

  except:

    print(f'Failed to fit exp on fatalities: {sys.exc_info()}')

    fatalities_exp, fatalities_exp_p, fatalities_exp_score = None, None, None



  params[key] = {

    'confirmed_cases_sigmoid_p': confirmed_cases_sigmoid_p,

    'confirmed_cases_sigmoid_score': confirmed_cases_sigmoid_score,

    'fatalities_sigmoid_p': fatalities_sigmoid_p,

    'fatalities_sigmoid_score': fatalities_sigmoid_score,

    'confirmed_cases_exp_p': confirmed_cases_exp_p,

    'confirmed_cases_exp_score': confirmed_cases_exp_score,

    'fatalities_exp_p': fatalities_exp_p,

    'fatalities_exp_score': fatalities_exp_score,

  }

#   if key not in keys_to_plot: continue



  validation_counts = get_counts_for_key(validation_data, key)



  fig = plt.figure(figsize=(12, 6))

  ax = fig.add_subplot(121)

  if confirmed_cases_sigmoid_score != None:

    ax.plot(

      all_dates,

      confirmed_cases_sigmoid,

      color=colors['light_blue'] if confirmed_cases_exp_score == None or confirmed_cases_exp_score > confirmed_cases_sigmoid_score else colors['light_gray'],

      linestyle=':',

      label=f'sigmoid {(confirmed_cases_sigmoid_score):.4f}'

    )

  if confirmed_cases_exp_score != None:

    ax.plot(

      all_dates,

      confirmed_cases_exp,

      color=colors['light_blue'] if confirmed_cases_sigmoid_score == None or confirmed_cases_sigmoid_score > confirmed_cases_exp_score else colors['light_gray'],

      label=f'exp {(confirmed_cases_exp_score):.4f}'

    )

  ax.plot(

    counts['Date'],

    counts['ConfirmedCases'],

    color=colors['very_dark_gray'],

    marker='.',

    linestyle='None'

  )

  ticks = all_dates[::13]

  label_from_tick = lambda tick: pd.to_datetime(tick).strftime('%b %d')

  tick_labels = list(map(label_from_tick, ticks))

  plt.xticks(ticks, tick_labels, rotation=20, horizontalalignment='right')

  plt.legend(loc='upper left')



  ax = fig.add_subplot(122)

  if fatalities_sigmoid_score != None:

    ax.plot(

      all_dates,

      fatalities_sigmoid,

      color=colors['light_blue'] if fatalities_exp_score == None or fatalities_exp_score > fatalities_sigmoid_score else colors['light_gray'],

      linestyle=':',

      label=f'sigmoid {(fatalities_sigmoid_score):.4f}'

    )

  if fatalities_exp_score != None:

    ax.plot(

      all_dates,

      fatalities_exp,

      color=colors['light_blue'] if fatalities_sigmoid_score == None or fatalities_sigmoid_score > fatalities_exp_score else colors['light_gray'],

      label=f'exp {(fatalities_exp_score):.4f}'

    )

  ax.plot(

    counts['Date'],

    counts['Fatalities'],

    color=colors['very_dark_gray'],

    marker='.',

    linestyle='None'

  )

  ticks = all_dates[::13]

  label_from_tick = lambda tick: pd.to_datetime(tick).strftime('%b %d')

  tick_labels = list(map(label_from_tick, ticks))

  plt.xticks(ticks, tick_labels, rotation=20, horizontalalignment='right')

  plt.legend(loc='upper left')



  plt.title(f'{key} results')

  plt.show()
def evaluate_model (p, metric, dates):

  model = 'sigmoid' if p[f'{metric}_sigmoid_score'] != None and p[f'{metric}_sigmoid_score'] < p[f'{metric}_exp_score'] else 'exp'

  model_params = p[f'{metric}_{model}_p']

  f = sigmoid if model == 'sigmoid' else exponential

  return np.maximum(np.round(f((dates - start_date) / np.timedelta64(1, 'D'), *model_params)), 0.0)



submission = test_data[['ForecastId', 'Date', 'ConfirmedCases', 'Fatalities', 'unique_area_key']].copy()

for key in submission['unique_area_key'].unique():

  ind = submission['unique_area_key'] == key

  dates = submission.loc[ind, 'Date']

  p = params[key]

  submission.loc[ind, 'ConfirmedCases'] = evaluate_model(p, 'confirmed_cases', dates)

  submission.loc[ind, 'Fatalities'] = evaluate_model(p, 'fatalities', dates)



submission_2 = submission[['ForecastId', 'ConfirmedCases', 'Fatalities']]

submission_2.astype('int32')

display(submission_2)

submission_2.to_csv('submission.csv', index=False)