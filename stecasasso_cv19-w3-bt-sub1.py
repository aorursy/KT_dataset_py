OPTIM_DAYS = 16 # Number of days to use for the optimisation evaluation

RUN_VAL = False

RUN_SUB = True

DATE_BORDER = '2020-04-08'



DEBUG = [

    # 'Italy',

    # 'France',

    # 'Hubei'

]
from joblib import Parallel, delayed

import multiprocessing

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

import os

from tqdm.auto import tqdm

from scipy.integrate import solve_ivp

from scipy.optimize import minimize

from sklearn.metrics import mean_squared_log_error, mean_squared_error
# Susceptible equation

def dS_dt(S, I, R_t, t_inf):

    return -(R_t / t_inf) * I * S





# Exposed equation

def dE_dt(S, E, I, R_t, t_inf, t_inc):

    return (R_t / t_inf) * I * S - (E / t_inc)





# Infected equation

def dI_dt(I, E, t_inc, t_inf):

    return (E / t_inc) - (I / t_inf)





# Hospialized equation

def dH_dt(I, C, H, t_inf, t_hosp, t_crit, m_a, f_a):

    return ((1 - m_a) * (I / t_inf)) + ((1 - f_a) * C / t_crit) - (H / t_hosp)





# Critical equation

def dC_dt(H, C, t_hosp, t_crit, c_a):

    return (c_a * H / t_hosp) - (C / t_crit)





# Recovered equation

def dR_dt(I, H, t_inf, t_hosp, m_a, c_a):

    return (m_a * I / t_inf) + (1 - c_a) * (H / t_hosp)





# Deaths equation

def dD_dt(C, t_crit, f_a):

    return f_a * C / t_crit





def SEIR_HCD_model(t, y, R_t, t_inc=2.9, t_inf=5.2, t_hosp=4, t_crit=10, m_a=0.8, c_a=0.1, f_a=0.3):

    """



    :param t: Time step for solve_ivp

    :param y: Previous solution or initial values

    :param R_t: Reproduction number

    :param t_inc: Average incubation period. Default 5.2 days

    :param t_inf: Average infectious period. Default 2.9 days

    :param t_hosp: Average time a patient is in hospital before either recovering or becoming critical. Default 4 days

    :param t_crit: Average time a patient is in a critical state (either recover or die). Default 14 days

    :param m_a: Fraction of infections that are asymptomatic or mild. Default 0.8

    :param c_a: Fraction of severe cases that turn critical. Default 0.1

    :param f_a: Fraction of critical cases that are fatal. Default 0.3

    :return:

    """

    if callable(R_t):

        reprod = R_t(t)

    else:

        reprod = R_t

        

    S, E, I, R, H, C, D = y

    

    S_out = dS_dt(S, I, reprod, t_inf)

    E_out = dE_dt(S, E, I, reprod, t_inf, t_inc)

    I_out = dI_dt(I, E, t_inc, t_inf)

    R_out = dR_dt(I, H, t_inf, t_hosp, m_a, c_a)

    H_out = dH_dt(I, C, H, t_inf, t_hosp, t_crit + t_hosp, m_a, f_a)

    C_out = dC_dt(H, C, t_hosp, t_crit + t_hosp, c_a)

    D_out = dD_dt(C, t_crit + t_hosp, f_a)

    return [S_out, E_out, I_out, R_out, H_out, C_out, D_out]
data_path = Path('/kaggle/input/covid19-global-forecasting-week-3/')



train = pd.read_csv(data_path / 'train.csv', parse_dates=['Date'])

test = pd.read_csv(data_path /'test.csv', parse_dates=['Date'])

submission = pd.read_csv(data_path /'submission.csv')



# Load the population data into lookup dicts

pop_info = pd.read_csv('/kaggle/input/covid19-population-data/population_data.csv')

country_pop = pop_info.query('Type == "Country/Region"')

province_pop = pop_info.query('Type == "Province/State"')

country_lookup = dict(zip(country_pop['Name'], country_pop['Population']))

province_lookup = dict(zip(province_pop['Name'], province_pop['Population']))



# Fix the Georgia State/Country confusion - probably a better was of doing this :)

train['Province_State'] = train['Province_State'].replace('Georgia', 'Georgia (State)')

test['Province_State'] = test['Province_State'].replace('Georgia', 'Georgia (State)')

province_lookup['Georgia (State)'] = province_lookup['Georgia']



train['Area'] = train['Province_State'].fillna(train['Country_Region'])

test['Area'] = test['Province_State'].fillna(test['Country_Region'])



# https://www.kaggle.com/c/covid19-global-forecasting-week-1/discussion/139172

train['ConfirmedCases'] = train.groupby('Area')['ConfirmedCases'].cummax()

train['Fatalities'] = train.groupby('Area')['Fatalities'].cummax()



# Remove the leaking data

train_full = train.copy()

valid = train[train['Date'] >= test['Date'].min()]

train = train[train['Date'] < test['Date'].min()]

VALID_START, VALID_END = valid['Date'].min(), valid['Date'].max()



# Split the test into public & private

test_public = test[test['Date'] <= DATE_BORDER]

test_private = test[test['Date'] > DATE_BORDER]

TEST_PUBLIC_START, TEST_PUBLIC_END = test_public['Date'].min(), test_public['Date'].max()

TEST_PRIVATE_START, TEST_PRIVATE_END = test_private['Date'].min(), test_private['Date'].max()



# Use a multi-index for easier slicing

train_full.set_index(['Area', 'Date'], inplace=True)

train.set_index(['Area', 'Date'], inplace=True)

valid.set_index(['Area', 'Date'], inplace=True)

test_public.set_index(['Area', 'Date'], inplace=True)

test_private.set_index(['Area', 'Date'], inplace=True)



# submission['ConfirmedCases'] = 0

# submission['Fatalities'] = 0

submission = submission.drop(['ConfirmedCases', 'Fatalities'], axis=1)



train_full.shape, train.shape, valid.shape, test_public.shape, test_private.shape, submission.shape

print(f'Public test: {TEST_PUBLIC_START} - {TEST_PUBLIC_END}')

print(f'Private test: {TEST_PRIVATE_START} - {TEST_PRIVATE_END}')

print(f'Validation: {VALID_START} - {VALID_END}')
submission = submission.merge(test[['ForecastId', 'Area', 'Date']], on='ForecastId', how='left')

submission.head(2)
popu = pop_info.copy()

popu.loc[(popu['Name'].str.contains('Georgia')) & (popu['Type'] == 'Province/State'), 'Name'] = 'Georgia (State)'
# Use a constant reproduction number

def eval_model_const(params, data, population, return_solution=False, forecast_days=0):

    R_0, t_hosp, t_crit, m, c, f = params

    N = population

    n_infected = data['ConfirmedCases'].iloc[0]

    max_days = len(data) + forecast_days

    initial_state = [(N - n_infected)/ N, 0, n_infected / N, 0, 0, 0, 0]

    args = (R_0, 5.6, 2.9, t_hosp, t_crit, m, c, f)

               

    sol = solve_ivp(SEIR_HCD_model, [0, max_days], initial_state, args=args, t_eval=np.arange(0, max_days))

    

    sus, exp, inf, rec, hosp, crit, deaths = sol.y

    

    y_pred_cases = np.clip(inf + rec + hosp + crit + deaths, 0, np.inf) * population

    y_true_cases = data['ConfirmedCases'].values

    y_pred_fat = np.clip(deaths, 0, np.inf) * population

    y_true_fat = data['Fatalities'].values

    

    optim_days = min(OPTIM_DAYS, len(data))  # Days to optimise for

    weights = 1 / np.arange(1, optim_days+1)[::-1]  # Recent data is more heavily weighted

    msle_cases = mean_squared_log_error(y_true_cases[-optim_days:], y_pred_cases[-optim_days:], weights)

    msle_fat = mean_squared_log_error(y_true_fat[-optim_days:], y_pred_fat[-optim_days:], weights)

    msle_final = np.mean([msle_cases, msle_fat])

    

    if return_solution:

        return msle_final, sol

    else:

        return msle_final
# Use a Hill decayed reproduction number

def eval_model_decay(params, data, population, return_solution=False, forecast_days=0):

    R_0, t_hosp, t_crit, m, c, f, k, L = params  

    N = population

    n_infected = data['ConfirmedCases'].iloc[0]

    max_days = len(data) + forecast_days

    

    # https://github.com/SwissTPH/openmalaria/wiki/ModelDecayFunctions   

    # Hill decay. Initial values: R_0=2.2, k=2, L=50

    def time_varying_reproduction(t): 

        return R_0 / (1 + (t/L)**k)

    

    initial_state = [(N - n_infected)/ N, 0, n_infected / N, 0, 0, 0, 0]

    args = (time_varying_reproduction, 5.6, 2.9, t_hosp, t_crit, m, c, f)

            

    sol = solve_ivp(SEIR_HCD_model, [0, max_days], initial_state, args=args, t_eval=np.arange(0, max_days))

    

    sus, exp, inf, rec, hosp, crit, deaths = sol.y

    

    y_pred_cases = np.clip(inf + rec + hosp + crit + deaths, 0, np.inf) * population

    y_true_cases = data['ConfirmedCases'].values

    y_pred_fat = np.clip(deaths, 0, np.inf) * population

    y_true_fat = data['Fatalities'].values

    

    optim_days = min(OPTIM_DAYS, len(data))  # Days to optimise for

    weights = 1 / np.arange(1, optim_days+1)[::-1]  # Recent data is more heavily weighted    

    msle_cases = mean_squared_log_error(y_true_cases[-optim_days:], y_pred_cases[-optim_days:], weights)

    msle_fat = mean_squared_log_error(y_true_fat[-optim_days:], y_pred_fat[-optim_days:], weights)

    msle_final = np.mean([msle_cases, msle_fat])

    

    if return_solution:

        return msle_final, sol

    else:

        return msle_final
def use_last_value(train_data, test_data):

    y_pred = train_data[['ConfirmedCases', 'Fatalities']].copy().reset_index()

    y_pred['R'] = 0.



    # Last value

    lv = pd.DataFrame(

        data={

            'ConfirmedCases': [y_pred.iloc[-1]['ConfirmedCases']],

            'Fatalities': [y_pred.iloc[-1]['Fatalities']],

            'R': [0.],

            'Date': [y_pred.iloc[-1]['Date']]

        }

    )



    dates_train = train_data.index.tolist()

    dates_test = [d for d in test_data.index.tolist() if d not in dates_train]

    dates_all = sorted(dates_train + dates_test)

    

    # Fill the test data

    for d in dates_test:

        _lv = lv.copy()

        _lv['Date'] = d

        y_pred = pd.concat([y_pred, _lv], ignore_index=True, sort=True)

    

    y_pred['Date'] = dates_all

    y_pred.set_index('Date', inplace=True)

    

    return y_pred
def plot_model_results(y_pred, train_data, valid_data=None, area=None):

    if area is None:

        area = 'unknown'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))

    

    ax1.set_title(f'{area} Confirmed Cases')

    ax2.set_title(f'{area} Fatalities')

    

    train_data['ConfirmedCases'].plot(label='Confirmed Cases (train)', color='g', ax=ax1)

    y_pred.loc[train_data.index, 'ConfirmedCases'].plot(label='Modeled Cases', color='r', ax=ax1)

    ax3 = y_pred['R'].plot(label='Reproduction number', color='c', linestyle='-', secondary_y=True, ax=ax1)

    ax3.set_ylabel("Reproduction number", fontsize=10, color='c');

        

    train_data['Fatalities'].plot(label='Fatalities (train)', color='g', ax=ax2)

    y_pred.loc[train_data.index, 'Fatalities'].plot(label='Modeled Fatalities', color='r', ax=ax2)

    

    if valid_data is not None:

        valid_data['ConfirmedCases'].plot(label='Confirmed Cases (valid)', color='g', linestyle=':', ax=ax1)

        valid_data['Fatalities'].plot(label='Fatalities (valid)', color='g', linestyle=':', ax=ax2)

        y_pred.loc[valid_data.index, 'ConfirmedCases'].plot(label='Modeled Cases (forecast)', color='r', linestyle=':', ax=ax1)

        y_pred.loc[valid_data.index, 'Fatalities'].plot(label='Modeled Fatalities (forecast)', color='r', linestyle=':', ax=ax2)

    else:

        y_pred.loc[:, 'ConfirmedCases'].plot(label='Modeled Cases (forecast)', color='r', linestyle=':', ax=ax1)

        y_pred.loc[:, 'Fatalities'].plot(label='Modeled Fatalities (forecast)', color='r', linestyle=':', ax=ax2)

        

    ax1.legend(loc='best')

    
def fit_model_public(area_name, 

                     initial_guess=[3.6, 4, 10, 0.8, 0.1, 0.3, 2, 50],

                     bounds=((1, 20), # R bounds

                             (0.5, 10), (2, 20), # transition time param bounds

                             (0.5, 1), (0, 1), (0, 1), (1, 5), (1, 100)), # fraction time param bounds

                     make_plot=True):

        

    train_data = train.loc[area_name].query('ConfirmedCases > 0')

    valid_data = valid.loc[area_name]

    test_data = test_public.loc[area_name]  

    

    try:

        population = province_lookup[area_name]

    except KeyError:

        population = country_lookup[area_name]

        

    cases_per_million = train_data['ConfirmedCases'].max() * 10**6 / population

    n_infected = train_data['ConfirmedCases'].iloc[0]

        

#     if cases_per_million < 1:

#         # print('Using last value')

#         return use_last_value(train_data, test_data)

                

    res_const = minimize(eval_model_const, initial_guess[:-2], bounds=bounds[:-2],

                         args=(train_data, population, False),

                         method='L-BFGS-B')

    

    res_decay = minimize(eval_model_decay, initial_guess, bounds=bounds,

                         args=(train_data, population, False),

                         method='L-BFGS-B')

    

    dates_all = train_data.index.append(test_data.index)

    dates_val = train_data.index.append(valid_data.index)

    

    

    # If using a constant R number is better, use that model

    if res_const.fun < res_decay.fun:

        msle, sol = eval_model_const(res_const.x, train_data, population, True, len(test_data))

        res = res_const

        R_t = pd.Series([res_const.x[0]] * len(dates_val), dates_val)

    else:

        msle, sol = eval_model_decay(res_decay.x, train_data, population, True, len(test_data))

        res = res_decay

        

        # Calculate the R_t values

        t = np.arange(len(dates_val))

        R_0, t_hosp, t_crit, m, c, f, k, L = res.x  

        R_t = pd.Series(R_0 / (1 + (t/L)**k), dates_val)

        

    sus, exp, inf, rec, hosp, crit, deaths = sol.y

    

    y_pred = pd.DataFrame({

        'ConfirmedCases': np.clip(inf + rec + hosp + crit + deaths, 0, np.inf) * population,

        'Fatalities': np.clip(deaths, 0, np.inf) * population,

        'R': R_t,

    }, index=dates_all)

    

    # Sanity check

    pred_max_cc, pred_max_f = y_pred['ConfirmedCases'].max(), y_pred['Fatalities'].max()

    obs_max_cc, obs_max_f = train_data['ConfirmedCases'].max(), train_data['Fatalities'].max()



    if (pred_max_cc < obs_max_cc) or (pred_max_f < obs_max_f):

        lv = use_last_value(train_data, test_data)

        if pred_max_cc < obs_max_cc:

            y_pred['ConfirmedCases'] = lv['ConfirmedCases']

        if pred_max_f < obs_max_f:

            y_pred['Fatalities'] = lv['Fatalities']      

    

    y_pred_valid = y_pred.iloc[len(train_data): len(train_data)+len(valid_data)]

    y_pred_test = y_pred.iloc[len(train_data):]

    y_true_valid = valid_data[['ConfirmedCases', 'Fatalities']]

        

    valid_msle_cases = np.sqrt(mean_squared_log_error(y_true_valid['ConfirmedCases'], y_pred_valid['ConfirmedCases']))

    valid_msle_fat = np.sqrt(mean_squared_log_error(y_true_valid['Fatalities'], y_pred_valid['Fatalities']))

    valid_msle = np.mean([valid_msle_cases, valid_msle_fat])

    

    if make_plot:

        print(f'Validation RMSLE: {valid_msle:0.5f}')

        print(f'R: {res.x[0]:0.3f}, t_hosp: {res.x[1]:0.3f}, t_crit: {res.x[2]:0.3f}, '

              f'm: {res.x[3]:0.3f}, c: {res.x[4]:0.3f}, f: {res.x[5]:0.3f}')

        plot_model_results(y_pred, train_data, valid_data, area=area_name)

        

#     # Put the forecast in the submission

#     forecast_ids = test_data['ForecastId']

#     submission.loc[forecast_ids, ['ConfirmedCases', 'Fatalities']] = y_pred_test[['ConfirmedCases', 'Fatalities']].values

    

    return (valid_msle, valid_msle_cases, valid_msle_fat), y_pred

            
# Fit a model on the full dataset (i.e. no validation)

def fit_model_private(area_name, 

                      initial_guess=[3.6, 4, 10, 0.8, 0.1, 0.3, 2, 50],

                      bounds=((1, 20), # R bounds

                              (0.5, 10), (2, 20), # transition time param bounds

                              (0.5, 1), (0, 1), (0, 1), (1, 5), (1, 100)), # fraction time param bounds

                      make_plot=True):

        

    train_data = train_full.loc[area_name].query('ConfirmedCases > 0')

    test_data = test_private.loc[area_name]

    

    try:

        population = province_lookup[area_name]

    except KeyError:

        population = country_lookup[area_name]

        

    cases_per_million = train_data['ConfirmedCases'].max() * 10**6 / population

    n_infected = train_data['ConfirmedCases'].iloc[0]

        

    if cases_per_million < 1:

        return use_last_value(train_data, test_data)

                

    res_const = minimize(eval_model_const, initial_guess[:-2], bounds=bounds[:-2],

                         args=(train_data, population, False),

                         method='L-BFGS-B')

    

    res_decay = minimize(eval_model_decay, initial_guess, bounds=bounds,

                         args=(train_data, population, False),

                         method='L-BFGS-B')

    

    dates_all = train_data.index.append(test_data.index)

    

    

    # If using a constant R number is better, use that model

    if res_const.fun < res_decay.fun:

        msle, sol = eval_model_const(res_const.x, train_data, population, True, len(test_data))

        res = res_const

        R_t = pd.Series([res_const.x[0]] * len(dates_all), dates_all)

    else:

        msle, sol = eval_model_decay(res_decay.x, train_data, population, True, len(test_data))

        res = res_decay

        

        # Calculate the R_t values

        t = np.arange(len(dates_all))

        R_0, t_hosp, t_crit, m, c, f, k, L = res.x  

        R_t = pd.Series(R_0 / (1 + (t/L)**k), dates_all)

        

    sus, exp, inf, rec, hosp, crit, deaths = sol.y

    

    y_pred = pd.DataFrame({

        'ConfirmedCases': np.clip(inf + rec + hosp + crit + deaths, 0, np.inf) * population,

        'Fatalities': np.clip(deaths, 0, np.inf) * population,

        'R': R_t,

    }, index=dates_all)

    

    # Sanity check

    pred_max_cc, pred_max_f = y_pred['ConfirmedCases'].max(), y_pred['Fatalities'].max()

    obs_max_cc, obs_max_f = train_data['ConfirmedCases'].max(), train_data['Fatalities'].max()

    if (pred_max_cc < obs_max_cc) or (pred_max_f < obs_max_f):

        lv = use_last_value(train_data, test_data)

        if pred_max_cc < obs_max_cc:

            y_pred['ConfirmedCases'] = lv['ConfirmedCases']

        if pred_max_f < obs_max_f:

            y_pred['Fatalities'] = lv['Fatalities']    

    

    y_pred_test = y_pred.iloc[len(train_data):]

    

    if make_plot:

        print(f'R: {res.x[0]:0.3f}, t_hosp: {res.x[1]:0.3f}, t_crit: {res.x[2]:0.3f}, '

              f'm: {res.x[3]:0.3f}, c: {res.x[4]:0.3f}, f: {res.x[5]:0.3f}')

        plot_model_results(y_pred, train_data, area=area_name)

        

#     # Put the forecast in the submission

#     forecast_ids = test_data['ForecastId']

#     submission.loc[forecast_ids, ['ConfirmedCases', 'Fatalities']] = y_pred_test[['ConfirmedCases', 'Fatalities']].values



    return y_pred

            
if DEBUG is None:

    DEBUG = []

for c in DEBUG:

    score, _ = fit_model_public(c)

    _ = fit_model_private(c)
def get_score(c):

    try:

        score, _ = fit_model_public(c, make_plot=False)

    except Exception as e:

        score = (np.nan, np.nan, np.nan)

    return {'Country': c, 'RMSLE': score[0], 'RMSLE_CASES': score[1], 'RMSLE_FATALITIES': score[2]}





if RUN_VAL:

    validation_scores = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(get_score)(c) for c in test_public.index.levels[0].values)

    validation_scores = pd.DataFrame(validation_scores)

    print(f'Mean validation score: {validation_scores["RMSLE"].mean():0.3f}')

    print(f'Cases validation score: {validation_scores["RMSLE_CASES"].mean():0.3f}')

    print(f'Fatalities validation score: {validation_scores["RMSLE_FATALITIES"].mean():0.3f}')          

    n_nans = validation_scores['RMSLE'].isnull().sum()

    print(f'NaNs = {n_nans}')

    # Find which areas are not being predicted well

    validation_scores = validation_scores.sort_values(by=['RMSLE'], ascending=False)

    print(validation_scores.head(20))
# Optim days = 210: Mean validation score: 0.826
def compute_scores(cc_true, cc_pred, f_true, f_pred):

    rmsle_cc = np.sqrt(mean_squared_log_error(cc_true, cc_pred))

    rmsle_f = np.sqrt(mean_squared_log_error(f_true, f_pred))

    rmsle_tot = np.mean([rmsle_cc, rmsle_f])

    return rmsle_tot, rmsle_cc, rmsle_f
def get_lb_preds(c):

    try:

        preds = fit_model_private(c, make_plot=False)

        preds = preds.reset_index().drop('R', axis=1)

        preds['Area'] = c

        return preds.merge(submission[['Area', 'Date', 'ForecastId']], on=['Area', 'Date'], how='left')

    except Exception as e:

        print(str(e))

        return pd.DataFrame()
if RUN_SUB:

    lb_preds = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(get_lb_preds)(c) for c in test_private.index.levels[0].values)

#     lb_preds = []

#     for c in tqdm(test_private.index.levels[0].values):

#         lb_preds.append(get_lb_preds(c))    
if RUN_SUB:

    df_preds = pd.concat(lb_preds)

    df_preds['ForecastId'] = df_preds['ForecastId'].fillna(-1).astype(int)

    # Prepare sub

    df_preds_test = df_preds[df_preds['Date'] >= TEST_PUBLIC_START].copy()

    df_preds_val = df_preds[(df_preds['Date'] >= VALID_START) & (df_preds['Date'] <= VALID_END)].copy()

    df_preds_val = df_preds_val.rename({'ConfirmedCases': 'ConfirmedCases_pred', 'Fatalities': 'Fatalities_pred'}, axis=1)

    df_preds_val = df_preds_val.merge(valid[['ConfirmedCases', 'Fatalities']], on=['Area', 'Date'], how='left')

    df_preds_val = df_preds_val.sort_values(['Area', 'Date'])

    print(VALID_START, VALID_END, df_preds['Date'].min(), df_preds['Date'].max())

    rmsle_tot, rmsle_cc, rmsle_f = compute_scores(

        df_preds_val['ConfirmedCases'], 

        df_preds_val['ConfirmedCases_pred'],

        df_preds_val['Fatalities'],

        df_preds_val['Fatalities_pred'],

    )

    print(rmsle_tot, rmsle_cc, rmsle_f)
print(df_preds_test.shape, df_preds_test.drop_duplicates(subset='ForecastId').shape)

df_preds_test = df_preds_test.merge(popu, left_on='Area', right_on='Name', how='left')

df_preds_test = df_preds_test.drop_duplicates(subset=['Area', 'Date'])

print(df_preds_test.shape)
print(df_preds_val.shape)

df_preds_val = df_preds_val.merge(popu, left_on='Area', right_on='Name', how='left')

df_preds_val = df_preds_val.drop_duplicates(subset=['Area', 'Date'])
# THR = 0.12

# best_thr, best_score = 1., rmsle_cc

# for thr in np.arange(0.000001, 0.0001, 0.000001):

#     sub = df_preds_val.copy()

#     sub['ConfirmedCases_pred'] = sub.apply(

#         lambda x: min(x['ConfirmedCases_pred'], x['Population'] * thr) if x['Population'] >= 100.E+06 else x['ConfirmedCases_pred'], axis=1

#     )    

#     score_tot, score_cc, score_f = compute_scores(

#             sub['ConfirmedCases'], 

#             sub['ConfirmedCases_pred'],

#             sub['Fatalities'],

#             sub['Fatalities_pred'],

#         )

#     if score_cc < best_score:

#         best_thr = float(thr)

#         best_score = float(score_cc)

#         print(thr, score_cc)

# df_preds_val['ConfirmedCases_pred'] = df_preds_val.apply(

#         lambda x: min(x['ConfirmedCases_pred'], x['Population'] * best_thr) if x['Population'] >= 100.E+06 else x['ConfirmedCases_pred'], axis=1

# )
# THR = 0.12

# best_thr = 1.

# for thr in np.arange(0.0001, 0.01, 0.0001):

#     sub = df_preds_val.copy()

#     sub['ConfirmedCases_pred'] = sub.apply(

#         lambda x: min(x['ConfirmedCases_pred'], x['Population'] * thr) if 10.E+06 <= x['Population'] < 100.E+06 else x['ConfirmedCases_pred'], axis=1

#     )    

#     score_tot, score_cc, score_f = compute_scores(

#             sub['ConfirmedCases'], 

#             sub['ConfirmedCases_pred'],

#             sub['Fatalities'],

#             sub['Fatalities_pred'],

#         )

#     if score_cc < best_score:

#         best_thr = float(thr)

#         best_score = float(score_cc)

#         print(thr, score_cc)

# df_preds_val['ConfirmedCases_pred'] = df_preds_val.apply(

#         lambda x: min(x['ConfirmedCases_pred'], x['Population'] * best_thr) if 10.E+06 <= x['Population'] < 100.E+06 else x['ConfirmedCases_pred'], axis=1

# )
# THR = 0.12

# best_thr = 1.

# for thr in np.arange(0.001, 1., 0.001):

#     sub = df_preds_val.copy()

#     sub['ConfirmedCases_pred'] = sub.apply(

#         lambda x: min(x['ConfirmedCases_pred'], x['Population'] * thr) if x['Population'] < 10.E+06 else x['ConfirmedCases_pred'], axis=1

#     )    

#     score_tot, score_cc, score_f = compute_scores(

#             sub['ConfirmedCases'], 

#             sub['ConfirmedCases_pred'],

#             sub['Fatalities'],

#             sub['Fatalities_pred'],

#         )

#     if score_cc < best_score:

#         best_thr = float(thr)

#         best_score = float(score_cc)

#         print(thr, score_cc)

# # df_preds_val['ConfirmedCases_pred'] = df_preds_val.apply(

# #         lambda x: min(x['ConfirmedCases_pred'], x['Population'] * best_thr) if x['Population'] < 10.E+06 else x['ConfirmedCases_pred'], axis=1

# # )
# THR = 0.12

# best_thr, best_score = 1., rmsle_f

# for thr in np.arange(0.01, 0.3, 0.01):

#     sub = df_preds_val.copy()

#     sub['Fatalities_pred'] = sub.apply(

#         lambda x: min(x['Fatalities_pred'], x['ConfirmedCases_pred'] * thr), axis=1

#     )

#     score_tot, score_cc, score_f = compute_scores(

#             sub['ConfirmedCases'], 

#             sub['ConfirmedCases_pred'],

#             sub['Fatalities'],

#             sub['Fatalities_pred'],

#         )

#     if score_f < best_score:

#         best_thr = thr

#         best_score = score_f

#         print(thr, score_f)

# df_preds_val['Fatalities_pred'] = df_preds_val.apply(

#         lambda x: min(x['Fatalities_pred'], x['ConfirmedCases_pred'] * best_thr), axis=1

# )
def cap_cases_val(x):

    if x['Population'] >= 100.E+06: 

        return min(x['ConfirmedCases_pred'], x['Population'] * 0.001)

    if x['Population'] >= 10.E+06: 

        return min(x['ConfirmedCases_pred'], x['Population'] * 0.0029)

    return min(x['ConfirmedCases_pred'], x['Population'] * 0.19)





def cap_cases(x):

    if x['Population'] >= 100.E+06: 

        return min(x['ConfirmedCases'], x['Population'] * 0.001)

    if x['Population'] >= 10.E+06: 

        return min(x['ConfirmedCases'], x['Population'] * 0.0029)

    return min(x['ConfirmedCases'], x['Population'] * 0.0029)
df_preds_val['ConfirmedCases_pred'] = df_preds_val.apply(cap_cases_val, axis=1)
df_preds_val['Fatalities_pred'] = df_preds_val['Fatalities_pred'].clip(0., df_preds_val['ConfirmedCases_pred'] * 0.26)
score_tot, score_cc, score_f = compute_scores(

        df_preds_val['ConfirmedCases'], 

        df_preds_val['ConfirmedCases_pred'],

        df_preds_val['Fatalities'],

        df_preds_val['Fatalities_pred'],

    )

print(score_tot, score_cc, score_f)
# Cap everything

df_preds_test['ConfirmedCases'] = df_preds_test.apply(cap_cases, axis=1)

df_preds_test['Fatalities'] = df_preds_test['Fatalities'].clip(0., df_preds_test['ConfirmedCases'] * 0.26)
if RUN_SUB:

    for country in test_private.index.levels[0].values:

        df_c = df_preds_val[df_preds_val['Area'] == country].copy()

        try:

            score_tot, score_cc, score_f = compute_scores(

                df_c['ConfirmedCases'], 

                df_c['ConfirmedCases_pred'],

                df_c['Fatalities'],

                df_c['Fatalities_pred'],

            )

            if score_tot > 0.5:

                print(f'{country}: {score_tot:.3f} {score_cc:.3f} {score_f:.3f}')

        except:

            print(f'No score for {country}')
if RUN_SUB:    

    submission = submission.merge(df_preds_test[['ForecastId', 'ConfirmedCases', 'Fatalities']], on='ForecastId', how='left')

submission.head()
if RUN_SUB:

    submission['ConfirmedCases'].isnull().sum(), submission['Fatalities'].isnull().sum()

    submission['Fatalities'] = submission['Fatalities'].fillna(0.)

    submission['ConfirmedCases'] = submission['ConfirmedCases'].fillna(0.)

    submission = submission[['ForecastId', 'ConfirmedCases', 'Fatalities']]

    submission.to_csv('submission.csv', index=False)