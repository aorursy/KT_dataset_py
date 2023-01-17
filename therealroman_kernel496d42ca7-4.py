# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from tqdm import tqdm_notebook as tqdm

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_log_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '/kaggle/input/covid19-global-forecasting-week-4/'

my_path = '/kaggle/input/week4-gg/'



subm = pd.read_csv(os.path.join(path, 'submission.csv'))

count = pd.read_csv(os.path.join(my_path, 'countries.csv'))

train = pd.read_csv(os.path.join(path, 'train.csv'))

test = pd.read_csv(os.path.join(path, 'test.csv'))
import pickle 



with open(os.path.join(my_path, 'dict_bst_ind.pickle'), 'rb') as f:

    dict_bst_ind = pickle.load(f)
new_train = pd.read_csv(os.path.join(my_path, 'cases_state.csv'))

new_train_2 = pd.read_csv(os.path.join(my_path, 'cases_country.csv'))
name_to_population_dict = {x:y for x, y in count[['ccse_name', 'population']].values}

train['iso'] = train['Country_Region'].astype(str) + '_' + train['Province_State'].astype(str)

test['iso'] = test['Country_Region'].astype(str) + '_' + test['Province_State'].astype(str)

new_train['iso'] = new_train['Country_Region'].astype(str) + '_' + new_train['Province_State'].astype(str)

new_train_2['iso'] = new_train_2['Country_Region'].astype(str) + '_' + pd.Series([np.nan] * len(new_train_2)).astype(str)

# train['population'] = train['Country_Region'].map(name_to_population_dict)
state_confirmed_dict = new_train.groupby('iso')['Confirmed'].last().to_dict()

state_death_dict = new_train.groupby('iso')['Deaths'].last().to_dict()



country_confirmed_dict = new_train_2.groupby('iso')['Confirmed'].last().to_dict()

country_death_dict = new_train_2.groupby('iso')['Deaths'].last().to_dict()
new_date_list = []

new_conf_list = []

new_fat_list = []

new_iso_list = []



for iso in train.iso.unique():

    if iso in state_confirmed_dict:

        new_conf_list += [state_confirmed_dict[iso]]

        new_fat_list += [state_death_dict[iso]]

        new_date_list += ['2020-04-15']

        new_iso_list += [iso]

    elif iso in country_confirmed_dict:

        new_conf_list += [country_confirmed_dict[iso]]

        new_fat_list += [country_death_dict[iso]]

        new_date_list += ['2020-04-15']

        new_iso_list += [iso]

    else:

        print('HAAAAAAAAAAI')
last_train = pd.DataFrame()

last_train['Date'] = new_date_list

last_train['ConfirmedCases'] = new_conf_list

last_train['Fatalities'] = new_fat_list

last_train['iso'] = new_iso_list



train = pd.concat([train, last_train], axis = 0).reset_index(drop = True)
def log_curve(x, x0, k, ymax):

    return ymax / (1 + np.exp(-k*(x-x0)))
valid_date = pd.to_datetime('2020-04-20')
target_cols = ['ConfirmedCases', 'Fatalities']



train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])



train = train.sort_values(['iso', 'Date']).reset_index(drop = True)
def get_train_piece(iso, valid_date):

    df = train[(train.iso == iso) & (train.Date < valid_date)].reset_index(drop = True)

    df = df[df['ConfirmedCases'] > 0].reset_index(drop = True)

    return df
def MALE(y_true, pred):

    return np.sqrt(mean_squared_log_error(y_true, pred))
def fit_predict(vals, len_test, best_ind, population):

    vals = vals[best_ind:]

   

    popt, pcov = curve_fit(log_curve, list(range(len(vals))), vals, 

                               bounds=([0,0, vals[-1]],[np.inf, np.inf, max(vals[-1] * 2, population * 0.01)]), 

                               p0=[10,0.3,vals[-1]], maxfev=1000000)

    pred = []

    for x in range(len(vals)-13, len(vals)-13 + len_test):

        pred += [log_curve(x, popt[0], popt[1], popt[2])]

    return pred
pred_df = pd.DataFrame()



for iso in tqdm(test.iso.unique()):

    train_df = get_train_piece(iso, valid_date)

    len_train = train_df.shape[0]

    

    test_df = test[test.iso == iso].reset_index(drop = True)

    len_test = test_df.shape[0]

    

    ans = pd.DataFrame()

    ans['ForecastId'] = test_df['ForecastId'].values

    population = name_to_population_dict.get(iso.split('_')[0], 1000000)

    

    

    if iso in dict_bst_ind['ConfirmedCases']:

        ans['ConfirmedCases'] = fit_predict(train_df['ConfirmedCases'].values, len_test, dict_bst_ind['ConfirmedCases'][iso], population)

    else:

        ans['ConfirmedCases'] = fit_predict(train_df['ConfirmedCases'].values, len_test, 0, population)

    if iso in dict_bst_ind['Fatalities']:

        ans['Fatalities'] = fit_predict(train_df['Fatalities'].values, len_test, dict_bst_ind['Fatalities'][iso], population)

    else:

        ans['Fatalities'] = fit_predict(train_df['Fatalities'].values, len_test, 0, population)



    

    pred_df = pd.concat([pred_df, ans], axis = 0).reset_index(drop = True)
pred_df
new_death = []



for x, y in pred_df[['ConfirmedCases', 'Fatalities']].values:

    if y / x > 0.2:

        new_death += [0.2 * x]

    else:

        new_death += [y]

pred_df['Fatalities'] = new_death
test
pred_df['Fatalities'].max()
pred_df.to_csv('submission.csv', index=False)