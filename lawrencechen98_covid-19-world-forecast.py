# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import scipy.optimize as opt

import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from tqdm import tqdm_notebook 

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error

import warnings; warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

train['Date'] = pd.to_datetime(train['Date'])

last_train_date = np.datetime64('2020-03-11')

eval_set = train[train['Date'] > last_train_date]



# overlap_days = 0

overlap_days = eval_set['Date'].unique()[-1] - last_train_date

overlap_days = int(overlap_days.astype('timedelta64[D]') / np.timedelta64(1, 'D'))

# train = train[train['Date'] <= last_train_date]



train
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

test['Date'] = pd.to_datetime(test['Date'])

test
sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

sub = sub.set_index('ForecastId', drop=True)

sub
def model(parameters, time):

    y_pred = parameters[0] * (1 - np.exp(-parameters[1] * (time - parameters[3]))) ** parameters[2]

    return np.nan_to_num(y_pred).clip(0, np.inf)



def residual(parameters, time, data):

    y_pred = model(parameters, time)

    return mean_squared_error(data, y_pred)
def fitModel(time, data, guess):

    params = opt.minimize(residual, guess, args=(time, data), bounds=opt.Bounds(0, np.inf, True), method='Nelder-Mead', tol=1e-7)

    return params.x
def trainModels(data):

    model_params = {}



    for country in tqdm_notebook(data['Country/Region'].unique()):

        country_data = data[data['Country/Region'] == country]

        for province in country_data['Province/State'].unique():  

            province_data = country_data[country_data['Province/State'] == province]

            if pd.isnull(province):

                province = None

                province_data = country_data

            for measure in ('ConfirmedCases', 'Fatalities'):

                filtered_data = province_data[measure]

                time_samples = len(filtered_data)

                try: 

                    start_date = filtered_data.nonzero()[0][0]

                    guess_offset = -np.log(filtered_data[start_date])

                except:

                    start_date = 0

                    guess_offset = 0

                guess_params = [filtered_data.max()*4, 0.1, 5, guess_offset]

                fit_params = fitModel(range(time_samples-start_date), filtered_data.iloc[start_date:], guess = guess_params)

                identifier = (country, province, measure)

                model_params[identifier] = {'params': fit_params, 'num_samples': time_samples, 'start_date': start_date}

    

    return model_params
model_params = trainModels(train)
def forecast(data, model_params, sub):

    model_predictions = {}

    for country in tqdm_notebook(data['Country/Region'].unique()):

        country_data = data[data['Country/Region'] == country]

        for province in country_data['Province/State'].unique():  

            province_data = country_data[country_data['Province/State'] == province]

            if pd.isnull(province):

                province = None

                province_data = country_data

            for measure in ('ConfirmedCases', 'Fatalities'):

                filtered_data = province_data['ForecastId']

                

                identifier = (country, province, measure)

                params = model_params[identifier]['params']

                num_samples = model_params[identifier]['num_samples']

                start_date = model_params[identifier]['start_date']

                

                predictions = model(params, range(num_samples-start_date-overlap_days, num_samples-start_date-overlap_days + len(filtered_data)))

                

                model_predictions[identifier] = predictions

                

                sub.loc[filtered_data, measure] = predictions

    return model_predictions
model_predictions = forecast(test, model_params, sub)
identifier = ('US', 'California', 'ConfirmedCases')



samples_train = train[(train['Country/Region'] == identifier[0]) ][identifier[2]]

samples_eval = eval_set[(eval_set['Country/Region'] == identifier[0]) ][identifier[2]]

if identifier[1] is not None:

    samples_train = train[(train['Country/Region'] == identifier[0]) & (train['Province/State'] == identifier[1])][identifier[2]]

    samples_eval = eval_set[(eval_set['Country/Region'] == identifier[0]) & (eval_set['Province/State'] == identifier[1])][identifier[2]]

params = model_params[identifier]['params']

print(params)

num_samples = model_params[identifier]['num_samples']

start_date = model_params[identifier]['start_date']

predictions = model_predictions[identifier]

plt.scatter(range(num_samples), samples_train)

# plt.scatter(range(num_samples, num_samples+len(samples_eval)), samples_eval)

plt.plot(range(start_date, num_samples + len(predictions)), model(params, range(num_samples-start_date + len(predictions))))
eval_with_id = pd.merge(eval_set, test, on=['Date', 'Country/Region', 'Province/State'])

sub = sub.replace([np.inf, -np.inf], np.nan)

sub = sub.fillna(0)

merged_eval = pd.merge(eval_with_id, sub, left_on='ForecastId', right_index=True)

merged_eval
# Evaluation score, not accurate if using full dataset where training and eval set overlap

score_confirmed = np.sqrt(mean_squared_log_error(merged_eval['ConfirmedCases_x'].values, merged_eval['ConfirmedCases_y']))

score_fatality = np.sqrt(mean_squared_log_error(merged_eval['Fatalities_x'].values, merged_eval['Fatalities_y']))

print(f'Confirmed Cases Score: {score_confirmed}\nFatality Score: {score_fatality}\nAverage Score: {np.mean([score_confirmed, score_fatality])}')
sub
sub.to_csv('submission.csv')