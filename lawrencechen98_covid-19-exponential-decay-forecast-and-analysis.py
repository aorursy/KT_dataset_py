import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import scipy.optimize as opt

import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from tqdm import tqdm_notebook 

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error

import warnings; warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

train['Date'] = pd.to_datetime(train['Date'])



# So we can keep track of when the training set ends, eval set starts, etc.

# Could be a better way to handle this, but this is spaghetti code from my iterations to handle offsets when predicting dates

first_test_date = np.datetime64('2020-03-19')

last_train_date = np.datetime64('2020-03-31')

eval_set = train[train['Date'] > first_test_date]



# overlap_days keeps track of how many training dates are also in the test dates, for offset purposes

overlap_days = last_train_date - first_test_date

overlap_days = int(overlap_days.astype('timedelta64[D]') / np.timedelta64(1, 'D'))



train = train[train['Date'] <= last_train_date]



train
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

test['Date'] = pd.to_datetime(test['Date'])

test
sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')

sub = sub.set_index('ForecastId', drop=True)

sub
def model(parameters, time):

    y_pred = parameters[0] * (1 - np.exp(-parameters[1] * (time - parameters[3])))**parameters[2] + np.maximum(time*parameters[4], 0)

    return np.nan_to_num(y_pred).clip(0, np.inf)



def residual(parameters, time, data):

    y_pred = model(parameters, time)

    return mean_squared_error(data, y_pred)
def fitModel(time, data, guess):

    params = opt.minimize(residual, guess, args=(time, data), method='Nelder-Mead', tol=1e-7)

    return params.x
def trainModels(data):

    model_params = {}



    for country in tqdm_notebook(data['Country_Region'].unique()):

        country_data = data[data['Country_Region'] == country]

        for province in country_data['Province_State'].unique():  

            province_data = country_data[country_data['Province_State'] == province]

            if pd.isnull(province):

                province = None

                province_data = country_data[country_data['Province_State'].isnull()]

            for measure in ('ConfirmedCases', 'Fatalities'):

                filtered_data = province_data[measure]

                time_samples = len(filtered_data)

                try: 

                    start_date = filtered_data.nonzero()[0][0]

                    # guess offset is log(day of first reported case), assuming exponential spread prior to proper test documented the case

                    guess_offset = -np.log(filtered_data[start_date])

                except:

                    start_date = 0

                    guess_offset = 0

                guess_params = [filtered_data.max()*2, 0.1, 5, guess_offset, 0]

                fit_params = fitModel(range(time_samples-start_date), filtered_data.iloc[start_date:], guess = guess_params)

                identifier = (country, province, measure)

                model_params[identifier] = {'params': fit_params, 'num_samples': time_samples, 'start_date': start_date, 'max_value': filtered_data.max()}

    

    return model_params
model_params = trainModels(train)
def forecast(data, model_params, sub):

    model_predictions = {}

    for country in tqdm_notebook(data['Country_Region'].unique()):

        country_data = data[data['Country_Region'] == country]

        for province in country_data['Province_State'].unique():  

            province_data = country_data[country_data['Province_State'] == province]

            if pd.isnull(province):

                province = None

                province_data = country_data[country_data['Province_State'].isnull()]

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
identifier = ('Taiwan*', None, 'ConfirmedCases')



samples_train = train[(train['Country_Region'] == identifier[0]) & train['Province_State'].isnull()][identifier[2]]

samples_eval = eval_set[(eval_set['Country_Region'] == identifier[0]) & eval_set['Province_State'].isnull()][identifier[2]]

if identifier[1] is not None:

    samples_train = train[(train['Country_Region'] == identifier[0]) & (train['Province_State'] == identifier[1])][identifier[2]]

    samples_eval = eval_set[(eval_set['Country_Region'] == identifier[0]) & (eval_set['Province_State'] == identifier[1])][identifier[2]]

params = model_params[identifier]['params']

print(params)

num_samples = model_params[identifier]['num_samples']

start_date = model_params[identifier]['start_date']

predictions = model_predictions[identifier]

plt.scatter(range(num_samples), samples_train)

# Plots the eval set points

# Note that when using full training set before final submission, the eval set overlaps with training set, so the graph is misleading in terms of fit

plt.scatter(range(num_samples-overlap_days, num_samples-overlap_days+len(samples_eval)), samples_eval)

plt.plot(range(start_date, num_samples + len(predictions)), model(params, range(num_samples-start_date + len(predictions))))
eval_with_id = pd.merge(eval_set, test, on=['Date', 'Country_Region', 'Province_State'])

sub = sub.replace([np.inf, -np.inf], np.nan)

sub = sub.fillna(0)

merged_eval = pd.merge(eval_with_id, sub, left_on='ForecastId', right_index=True)

merged_eval
# Evaluation score, not accurate if using full dataset where training and eval set overlap

score_confirmed = np.sqrt(mean_squared_log_error(merged_eval['ConfirmedCases_x'].values, merged_eval['ConfirmedCases_y'].values))

score_fatality = np.sqrt(mean_squared_log_error(merged_eval['Fatalities_x'].values, merged_eval['Fatalities_y'].values))

print(f'Confirmed Cases Score: {score_confirmed}\nFatality Score: {score_fatality}\nAverage Score: {np.mean([score_confirmed, score_fatality])}')
sub.to_csv('submission.csv')

sub
def p2f(x):

    try:

        return float(x.strip('%'))/100

    except:

        return None

    

def s2f(x):

    try:

        return float(x)

    except:

        return None



pop_set = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv', converters={'Med. Age': s2f, 'Urban Pop %': p2f})

pop_set.rename(columns={ pop_set.columns[0]: "Country" , pop_set.columns[1]: "Population" , pop_set.columns[4]: "Density", pop_set.columns[8]: 'Median Age', pop_set.columns[9]: "Urban Pop"}, inplace = True)

pop_set = pop_set.iloc[:, [0, 1, 4, 8, 9]]

pop_set = pop_set.replace(regex={

    'Czech Republic (Czechia)': 'Czechia', 

    "Côte d'Ivoire": "Cote d'Ivoire", 

    'St. Vincent & Grenadines': 'Saint Vincent and the Grenadines', 

    'Saint Kitts & Nevis': 'Saint Kitts and Nevis', 

    'Taiwan': 'Taiwan*', 

    'South Korea': 'Korea, South', 

    'United States': 'US'})

pop_set
freedom_set = pd.read_csv('/kaggle/input/cato-2017-human-freedom-index/cato_2017_hfi_by_year_summary.csv')

freedom_set = freedom_set[freedom_set['Year'] == 2015]

freedom_set.rename(columns={ freedom_set.columns[2]: "Country" , freedom_set.columns[3]: "Personal Freedom" , freedom_set.columns[4]: "Economic Freedom", freedom_set.columns[5]: 'Human Freedom'}, inplace = True)

freedom_set = freedom_set.iloc[:, [2, 3, 4, 5]]

freedom_set = freedom_set.replace(regex={

    'Czech Republic': 'Czechia', 

    'Taiwan': 'Taiwan*', 

    'Korea, Republic of': 'Korea, South', 

    'United States': 'US'})

freedom_set
country_set = pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv', decimal=",")

country_set.rename(columns={ country_set.columns[7]: "Infant Mortality" , country_set.columns[8]: "GDP" , country_set.columns[9]: "Literacy"}, inplace = True)

country_set = country_set.iloc[:, [0,7,8,9]]

country_set['Country'] = country_set['Country'].str.strip()

country_set = country_set.replace(regex={

    'Czech Republic': 'Czechia', 

    'Taiwan': 'Taiwan*', 

    'Korea, Republic of': 'Korea, South', 

    'United States': 'US'})

country_set
from scipy import stats

def analyzeParams(model_params, index, measurement, external_set, external_column, threshold= 1000, label=False, xmin=None, xmax=None, ymin=None, ymax=None):

    

    external_vals = []

    params_vals = []

    names = []

    

    for identifier in model_params:

        

        if identifier[0] not in ['US', 'China'] and identifier[2] == measurement and model_params[identifier]['max_value'] > threshold:

            try:

                val = external_set.loc[external_set['Country'] == identifier[0], external_column].values[0]

                if np.isnan(val):

                    continue

                external_vals.append(val)

                if identifier[1] is None:

                    names.append(identifier[0])

                else:

                    names.append(identifier[1] + ', ' + identifier[0])



                params = model_params[identifier]['params']

                params_vals.append(params[index])

            except:

                continue

    



    R = np.corrcoef(external_vals, params_vals)

    plt.figure(figsize=(8, 6))

    plt.scatter(external_vals, params_vals)

    plt.xlim(xmin, xmax)

    plt.ylim(ymin, ymax)

    plt.title("Correlation R = " + str(R[0, 1]))



    if label:

        for i, name in enumerate(names):

            if (xmax is None or external_vals[i] <= xmax) and (xmin is None or external_vals[i] >= xmin) and (ymax is None or yparams_vals[i] <= ymax) and (ymin is None or params_vals[i] >= ymin):

                plt.text(external_vals[i], params_vals[i], name, size=12)

    
analyzeParams(model_params, 1, 'ConfirmedCases', pop_set, 'Density', 500, True)
analyzeParams(model_params, 1, 'Fatalities', pop_set, 'Median Age', 10, True)
analyzeParams(model_params, 1, 'ConfirmedCases', pop_set, 'Median Age', 500, True)
analyzeParams(model_params, 1, 'ConfirmedCases', freedom_set, 'Economic Freedom', 500, True)
analyzeParams(model_params, 1, 'ConfirmedCases', freedom_set, 'Personal Freedom', 500, True)
analyzeParams(model_params, 1, 'ConfirmedCases', freedom_set, 'Human Freedom', 500, True)
analyzeParams(model_params, 1, 'Fatalities', country_set, 'Infant Mortality', 10, True)
analyzeParams(model_params, 1, 'ConfirmedCases', country_set, 'Literacy', 500, True)
analyzeParams(model_params, 1, 'ConfirmedCases', country_set, 'GDP', 500, True)