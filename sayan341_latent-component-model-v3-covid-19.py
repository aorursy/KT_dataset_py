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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.tsa.arima_model as arima

from sklearn.linear_model import LinearRegression

from sklearn import linear_model
import warnings

warnings.filterwarnings('ignore')

pd.set_option("display.max_rows", None, "display.max_columns", None)
training_data_original = pd.read_csv('/kaggle/input/inputs2/train.csv')

training_data_original['Date'] = pd.to_datetime(training_data_original['Date'])

training_data = training_data_original[~training_data_original['Country_Region'].isin(['Diamond Princess', 'MS Zaandam'])]

# training_data = training_data.set_index('Date')



countries = training_data['Country_Region'].unique()

countries_main = ['China', 'US', 'Australia', 'Canada']



states = training_data['Province_State'].unique()



training_data[training_data['Country_Region'] == 'Belize']
plt.figure(figsize = (16, 7))

ax = plt.gca()

for country in countries:

    if country in ['China', 'US', 'Australia', 'Canada']:

        country_data = training_data[training_data['Country_Region'] == country]

        country_data = country_data.groupby(['Date']).sum()

        country_data.reset_index().plot(kind = 'line', x='Date', y='ConfirmedCases', ax=ax, label = country)
state_metadata = pd.read_excel('/kaggle/input/externaldata2/states.xlsx')

country_metadata = pd.read_excel('/kaggle/input/externaldata2/countries.xlsx')
country_metadata.head()
def data_preparation(training_data, countries, states, country_metadata, state_metadata, n_days_case, 

                     n_days_fatal, min_num_cases = 2):

    training_data_trun = training_data[training_data['ConfirmedCases'] >= min_num_cases]

    conf_cases_dict = {}

    fatal_dict = {}

    

    for country in countries:

        if country not in countries_main:

            training_data_trun_loc = training_data_trun[(training_data_trun['Country_Region'] == country) & (pd.isnull(training_data_trun.Province_State))]

            training_data_trun_loc = training_data_trun_loc.groupby(['Date']).sum()

            training_data_trun_loc = training_data_trun_loc.sort_values(by = 'Date')

            if len(training_data_trun_loc['ConfirmedCases'].values) >= n_days_case:

                conf_cases_dict[country] = training_data_trun_loc['ConfirmedCases'].values[:n_days_case] / country_metadata[country_metadata['Countries'] == country]['Population'].values[0]

            if len(training_data_trun_loc['Fatalities'].values) >= n_days_fatal:

                fatal_dict[country] = training_data_trun_loc['Fatalities'].values[:n_days_fatal] #/ country_metadata[country_metadata['Countries'] == country]['Population'].values[0]

            

    for state in states:

        training_data_trun_loc = training_data_trun[training_data_trun['Province_State'] == state]

        training_data_trun_loc = training_data_trun_loc.groupby(['Date']).sum()

        training_data_trun_loc = training_data_trun_loc.sort_values(by = 'Date')

        if len(training_data_trun_loc['ConfirmedCases'].values) >= n_days_case:

            conf_cases_dict[state] = training_data_trun_loc['ConfirmedCases'].values[:n_days_case] / state_metadata[state_metadata['States'] == state]['Population'].values[0]

        if len(training_data_trun_loc['Fatalities'].values) >= n_days_fatal:

            fatal_dict[state] = training_data_trun_loc['Fatalities'].values[:n_days_fatal] #/ state_metadata[state_metadata['States'] == state]['Population'].values[0]



    return pd.DataFrame(conf_cases_dict), pd.DataFrame(fatal_dict)
def fts_training(input_df, rank = 3):

    

    matrix = input_df.values

    

    u, s, v = np.linalg.svd(matrix, full_matrices=False)

        

    scores = np.matmul(u[:, :rank], np.diag(s[:rank]))

    pcs = v[:rank, :]

    

    return scores, pcs
def forecast_trajectories(training_data, countries, states, country_metadata, 

                          state_metadata, loc = None, n_days_case = 30, n_days_fatal = 10, 

                          forecast_days = 10, min_num_cases = 2, model_type = 'ARIMA',

                          components_modeling = True, rank = 3):



    from pykalman import KalmanFilter

    kf = KalmanFilter(initial_state_mean = np.zeros(rank).tolist(), n_dim_obs = rank)

    

    # Training Data

    conf_cases_df, fatal_df = data_preparation(training_data, countries, states, 

                                               country_metadata, state_metadata, 

                                               n_days_case = n_days_case, 

                                               n_days_fatal = n_days_case,

                                               min_num_cases = min_num_cases)

    pred_countries = conf_cases_df.columns.tolist()

    pred_countries_fatal = fatal_df.columns.tolist()

    

    # Training Features

    conf_cases_exog_df, fatal_exog_df = data_preparation(training_data, countries, states, 

                                                         country_metadata, state_metadata, 

                                                         n_days_case = n_days_case + forecast_days, 

                                                         n_days_fatal = n_days_case + forecast_days,

                                                         min_num_cases = min_num_cases)

    

    scores_exog, pcs_exog = fts_training(conf_cases_exog_df, rank = rank)

    

    if len(scores_exog) > 0:

        scores_exog = kf.em(scores_exog).smooth(scores_exog)[0]

    

    scores, pcs = fts_training(conf_cases_df, rank = rank)



    forecasted_scores = []

    idx = 0

    for score in scores.T:

        if components_modeling:

            exog = scores_exog[:n_days_case, idx] if len(scores_exog) > 0 else None

            pred_exog = scores_exog[n_days_case:, idx] if len(scores_exog) > 0 else None

            y = score

        else:

            exog = scores_exog[:n_days_case, :] if len(scores_exog) > 0 else None

            pred_exog = scores_exog[n_days_case:, :] if len(scores_exog) > 0 else None

            y = conf_cases_df[loc].values

        try:

            model = arima.ARIMA(endog = y, exog = exog, order = (4, 1, 0)).fit(

                                seasonal = False,

                                trace = False,

                                method = 'css',

                                solver = 'bfgs',

                                error_action = 'ignore',

                                setpwise_fit = True,

                                warn_convergence = True,

                                disp = False)

        except:

            try:

                model = arima.ARIMA(endog = y, exog = exog, order = (3, 1, 0)).fit(

                                    seasonal = False,

                                    trace = False,

                                    method = 'css',

                                    solver = 'bfgs',

                                    error_action = 'ignore',

                                    setpwise_fit = True,

                                    warn_convergence = True,

                                    disp = False)

            except:

                try:

                    model = arima.ARIMA(endog = y, exog = exog, order = (2, 1, 0)).fit(

                                        seasonal = False,

                                        trace = False,

                                        method = 'css',

                                        solver = 'bfgs',

                                        error_action = 'ignore',

                                        setpwise_fit = True,

                                        warn_convergence = True,

                                        disp = False)

                except:

                    model = arima.ARIMA(endog = y, exog = exog, order = (1, 0, 0)).fit(

                                        seasonal = False,

                                        trace = False,

                                        method = 'css',

                                        solver = 'bfgs',

                                        error_action = 'ignore',

                                        setpwise_fit = True,

                                        warn_convergence = True,

                                        disp = False)

        if not components_modeling:

            pred_traj = model.forecast(steps = forecast_days, 

                                                    alpha = 0.001, 

                                                    exog = pred_exog)[0]

            break                

        else:    

            forecasted_scores.append(model.forecast(steps = forecast_days, 

                                                    alpha = 0.001, 

                                                    exog = pred_exog)[0].tolist())

            idx = idx + 1



    if components_modeling:

        pred_traj = np.matmul(np.array(forecasted_scores).T, pcs)

        pred_traj_df = pd.DataFrame(pred_traj, columns = pred_countries)

        for loc in pred_countries:

            if loc in country_metadata['Countries'].values.tolist():

                pred_traj_df[loc] = country_metadata[country_metadata['Countries'] 

                                                         == loc]['Population'].values[0] * pred_traj_df[loc]

            if loc in state_metadata['States'].values.tolist():

                pred_traj_df[loc] = state_metadata[state_metadata['States'] 

                                                         == loc]['Population'].values[0] * pred_traj_df[loc]

    else:

        pred_traj_df = pd.DataFrame()

        if loc in countries:

            pred_traj_df[loc] = country_metadata[country_metadata['Countries'] 

                                                     == loc]['Population'].values[0] * pred_traj

        elif loc in states:

            pred_traj_df[loc] = state_metadata[state_metadata['States'] 

                                                     == loc]['Population'].values[0] * pred_traj



    #####################################################

    

    fatal_scores_exog, fatal_pcs_exog = fts_training(fatal_exog_df, rank = rank)

            

    if len(fatal_pcs_exog) > 0:

        fatal_scores_exog = kf.em(fatal_scores_exog).smooth(fatal_scores_exog)[0]

    

    fatal_scores, fatal_pcs = fts_training(fatal_df, rank = rank)



    forecasted_fatal_scores = []

    idx = 0

    for fatal_score in fatal_scores.T:

        if components_modeling:

            exog = fatal_scores_exog[:n_days_fatal, idx] if len(fatal_scores_exog) > 0 else None

            pred_exog = fatal_scores_exog[n_days_fatal:, idx] if len(fatal_scores_exog) > 0 else None

            y = fatal_score

        else:

            exog = fatal_scores_exog[:n_days_fatal, :] if len(fatal_scores_exog) > 0 else None

            pred_exog = fatal_scores_exog[n_days_fatal:, :] if len(fatal_scores_exog) > 0 else None

            y = fatal_df[loc].values   

        try:

            model = arima.ARIMA(endog = y, exog = exog, order = (4, 1, 0)).fit(

                                seasonal = False,

                                trace = False,

                                method = 'css',

                                solver = 'bfgs',

                                error_action = 'ignore',

                                setpwise_fit = True,

                                warn_convergence = True,

                                disp = False)

        except:

            try:

                model = arima.ARIMA(endog = y, exog = exog, order = (3, 1, 0)).fit(

                                    seasonal = False,

                                    trace = False,

                                    method = 'css',

                                    solver = 'bfgs',

                                    error_action = 'ignore',

                                    setpwise_fit = True,

                                    warn_convergence = True,

                                    disp = False)

            except:

                try:

                    model = arima.ARIMA(endog = y, exog = exog, order = (2, 1, 0)).fit(

                                        seasonal = False,

                                        trace = False,

                                        method = 'css',

                                        solver = 'bfgs',

                                        error_action = 'ignore',

                                        setpwise_fit = True,

                                        warn_convergence = True,

                                        disp = False)

                except:

                    model = arima.ARIMA(endog = y, exog = exog, order = (1, 0, 0)).fit(

                                        seasonal = False,

                                        trace = False,

                                        method = 'css',

                                        solver = 'bfgs',

                                        error_action = 'ignore',

                                        setpwise_fit = True,

                                        warn_convergence = True,

                                        disp = False)

        if not components_modeling:

            fatal_pred_traj = model.forecast(steps = forecast_days, 

                                                    alpha = 0.001, 

                                                    exog = pred_exog)[0]

            break                

        else:    

            forecasted_fatal_scores.append(model.forecast(steps = forecast_days, 

                                                          alpha = 0.001, 

                                                          exog = pred_exog)[0].tolist())

            idx = idx + 1



    if components_modeling:

        fatal_pred_traj = np.matmul(np.array(forecasted_fatal_scores).T, fatal_pcs)

        fatal_pred_traj_df = pd.DataFrame(fatal_pred_traj, columns = pred_countries_fatal)

    else:

        fatal_pred_traj_df = pd.DataFrame()

        if loc in countries:

            fatal_pred_traj_df[loc] = country_metadata[country_metadata['Countries'] 

                                                     == loc]['Population'].values[0] * fatal_pred_traj

        elif loc in states:

            fatal_pred_traj_df[loc] = state_metadata[state_metadata['States'] 

                                                     == loc]['Population'].values[0] * fatal_pred_traj

    

    return pred_traj_df, fatal_pred_traj_df
pred_traj_df, fatal_pred_traj_df = forecast_trajectories(training_data, countries, states, country_metadata, state_metadata, 

                                                         loc = 'New York', n_days_case = 29, n_days_fatal = 29, 

                                                         forecast_days = 30, rank = 3, min_num_cases = 25,

                                                         components_modeling = True)
pred_traj_df
def generate_prediction(test_data, training_data, countries_main, countries, states, min_num_cases):

    import math

    

    for country in countries:

        print(country)

        if country not in countries_main and country not in excl_list:

            test_loc_df = test_data[(test_data['Country_Region'] == country) & (pd.isnull(test_data.Province_State))].reset_index()

            train_loc_df = training_data[(training_data['Country_Region'] == country) & (pd.isnull(training_data.Province_State))].reset_index()

            test_start = test_loc_df['Date'][0]

            test_end = test_loc_df['Date'][len(test_loc_df) - 1]

            train_end = train_loc_df['Date'][len(train_loc_df) - 1]

            test_loc_df.loc[((test_loc_df['Date'] >= test_start)) & (test_loc_df['Date'] <= train_end), 'ConfirmedCases'] = train_loc_df[(train_loc_df['Date'] >= test_start) & (train_loc_df['Date'] <= train_end)]['ConfirmedCases'].values

            test_loc_df.loc[((test_loc_df['Date'] >= test_start)) & (test_loc_df['Date'] <= train_end), 'Fatalities'] = train_loc_df[(train_loc_df['Date'] >= test_start) & (train_loc_df['Date'] <= train_end)]['Fatalities'].values

            effective_df = train_loc_df[train_loc_df['ConfirmedCases'] >= min_num_cases]

            forecast_days = int((test_end - train_end).days)

            min_num_cases_temp = min_num_cases

            if len(effective_df) > 0:

                while min_num_cases_temp > 1:

                    try:

                        effective_train_start = train_loc_df[train_loc_df['ConfirmedCases'] >= min_num_cases_temp].reset_index()['Date'][0] 

                        n_days_case = int((train_end - effective_train_start).days) + 1



                        pred_df, fatal_pred_df = forecast_trajectories(training_data, countries, states, 

                                                                       country_metadata, state_metadata, loc = country,

                                                                       n_days_case = n_days_case, n_days_fatal = n_days_case, 

                                                                       forecast_days = forecast_days, min_num_cases = min_num_cases_temp,

                                                                       rank = 3)



                        test_loc_df.loc[test_loc_df['Date'] > train_end, 'ConfirmedCases'] = np.maximum.accumulate(pred_df[country].values).astype(int)

                        test_data.loc[(test_data['Country_Region'] == country) & (pd.isnull(test_data.Province_State)), 'ConfirmedCases'] = test_loc_df['ConfirmedCases'].values



                        test_loc_df.loc[test_loc_df['Date'] > train_end, 'Fatalities'] = np.maximum.accumulate(fatal_pred_df[country].values).astype(int)

                        test_data.loc[(test_data['Country_Region'] == country) & (pd.isnull(test_data.Province_State)), 'Fatalities'] = test_loc_df['Fatalities'].values

                        break

                    except:

                        min_num_cases_temp = math.floor(min_num_cases_temp / 2)

                        continue

    for state in states:

        print(state)

        if str(state) not in excl_list:

            test_loc_df = test_data[(test_data['Province_State'] == state)].reset_index()

            train_loc_df = training_data[(training_data['Province_State'] == state)].reset_index()

            test_start = test_loc_df['Date'][0]

            test_end = test_loc_df['Date'][len(test_loc_df) - 1]

            train_end = train_loc_df['Date'][len(train_loc_df) - 1]

            test_loc_df.loc[((test_loc_df['Date'] >= test_start)) & (test_loc_df['Date'] <= train_end), 'ConfirmedCases'] = train_loc_df[(train_loc_df['Date'] >= test_start) & (train_loc_df['Date'] <= train_end)]['ConfirmedCases'].values

            test_loc_df.loc[((test_loc_df['Date'] >= test_start)) & (test_loc_df['Date'] <= train_end), 'Fatalities'] = train_loc_df[(train_loc_df['Date'] >= test_start) & (train_loc_df['Date'] <= train_end)]['Fatalities'].values

            effective_df = train_loc_df[train_loc_df['ConfirmedCases'] >= min_num_cases]

            forecast_days = int((test_end - train_end).days)

            min_num_cases_temp = min_num_cases

            if len(effective_df) > 0:

                while min_num_cases_temp > 1:

                    try:

                        effective_train_start = train_loc_df[train_loc_df['ConfirmedCases'] >= min_num_cases_temp].reset_index()['Date'][0] 

                        n_days_case = int((train_end - effective_train_start).days) + 1



                        pred_df, fatal_pred_df = forecast_trajectories(training_data, countries, states, 

                                                                       country_metadata, state_metadata, loc = state,

                                                                       n_days_case = n_days_case, n_days_fatal = n_days_case, 

                                                                       forecast_days = forecast_days, min_num_cases = min_num_cases_temp,

                                                                       rank = 3)



                        test_loc_df.loc[test_loc_df['Date'] > train_end, 'ConfirmedCases'] = np.maximum.accumulate(pred_df[state].values).astype(int)

                        test_data.loc[test_data['Province_State'] == state, 'ConfirmedCases'] = test_loc_df['ConfirmedCases'].values



                        test_loc_df.loc[test_loc_df['Date'] > train_end, 'Fatalities'] = np.maximum.accumulate(fatal_pred_df[state].values).astype(int)

                        test_data.loc[test_data['Province_State'] == state, 'Fatalities'] = test_loc_df['Fatalities'].values

                        break

                    except:

                        min_num_cases_temp = math.floor(min_num_cases_temp / 2)

                        continue

    return test_data
test_data = pd.read_csv('/kaggle/input/inputs2/test.csv')

test_data['Date'] = pd.to_datetime(test_data['Date'])
test_data['ConfirmedCases'] = None

test_data['Fatalities'] = None
excl_list = ['Belize', 'Botswana', 'Diamond Princess', 'MS Zaandam', 'Angola',

             'Turks and Caicos Islands', 'Burma', 'Burundi', 'Chad', 'Eritrea',

            'Grenada', 'Guinea-Bissau', 'Holy See', 'Kosovo',

            'Laos', 'Libya', 'Mali', 'Mozambique', 'Saint Kitts and Nevis', 'Somalia',

             'Syria', 'nan', 'Saint Barthelemy', 'Virgin Islands', 'Montserrat',

            'Saint Vincent and the Grenadines', 'Sierra Leone', 'Northwest Territories',

            'Yukon', 'Anguilla', 'British Virgin Islands', 'Papua New Guinea', 'Bhutan',

            'Congo (Brazzaville)', 'Gabon', 'Guinea', 'Guyana', 'Haiti', 'Namibia',

             'Saint Lucia', 'Seychelles', 'Curacao', 'Cayman Islands',

            'Central African Republic', 'Liberia', 'Mauritania', 'Nepal',

            'Nicaragua', 'Sudan']
predictions = generate_prediction(test_data, training_data, countries_main, countries, states, min_num_cases = 25)
missing_countries = predictions[(predictions.ConfirmedCases.isnull()) & (pd.isnull(predictions.Province_State))]['Country_Region'].unique().tolist()

missing_states = predictions[(predictions.ConfirmedCases.isnull()) & (pd.notnull(predictions.Province_State))]['Province_State'].unique().tolist()
def fill_excl_pred(predictions, training_data_original, test_data):

    

    for country in missing_countries:

        print(country)

        pred_loc_df = predictions[predictions['Country_Region'] == country]

        train_loc_df = training_data_original[training_data_original['Country_Region'] == country]



        series_comf_cases = train_loc_df['ConfirmedCases'].values

        series_fatal = train_loc_df['Fatalities'].values



        test_start = test_data[test_data['Country_Region'] == country]['Date'].values[0]

        series_comf_cases_test = training_data_original[(training_data_original['Country_Region'] == country) & (training_data_original['Date'] >= test_start)]['ConfirmedCases']

        series_fatal_test = training_data_original[(training_data_original['Country_Region'] == country) & (training_data_original['Date'] >= test_start)]['Fatalities']



        if len(series_comf_cases) > 0:

            regressor = LinearRegression() 



            regressor.fit(np.arange(len(series_comf_cases_test)).reshape(-1, 1), series_comf_cases_test)

            comf_cases_pred = regressor.predict(np.arange(13, 43).reshape(-1, 1))

            regressor.fit(np.arange(len(series_fatal_test)).reshape(-1, 1), series_fatal_test)

            fatal_pred = regressor.predict(np.arange(13, 43).reshape(-1, 1))

        else:

            comf_cases_pred = []

            fatal_pred = []



        conf_cases_loc = np.concatenate((series_comf_cases_test, comf_cases_pred), axis=0)

        fatal_loc = np.concatenate((series_fatal_test, fatal_pred), axis=0)



        predictions.loc[predictions['Country_Region'] == country, 'ConfirmedCases'] = conf_cases_loc.astype(int)

        predictions.loc[predictions['Country_Region'] == country, 'Fatalities'] = fatal_loc.astype(int)



    for state in missing_states:

        print(state)

        pred_loc_df = predictions[predictions['Province_State'] == state]

        train_loc_df = training_data_original[training_data_original['Province_State'] == state]

        series_comf_cases = train_loc_df['ConfirmedCases'].values

        series_fatal = train_loc_df['Fatalities'].values



        test_start = test_data[test_data['Province_State'] == state]['Date'].values[0]

        series_comf_cases_test = training_data_original[(training_data_original['Province_State'] == state) & (training_data_original['Date'] >= test_start)]['ConfirmedCases']

        series_fatal_test = training_data[(training_data_original['Province_State'] == state) & (training_data_original['Date'] >= test_start)]['Fatalities']



        regressor = LinearRegression() 



        regressor.fit(np.arange(len(series_comf_cases_test)).reshape(-1, 1), series_comf_cases_test)

        comf_cases_pred = regressor.predict(np.arange(13, 43).reshape(-1, 1))



        regressor.fit(np.arange(len(series_fatal_test)).reshape(-1, 1), series_fatal_test)

        fatal_pred = regressor.predict(np.arange(13, 43).reshape(-1, 1))



        conf_cases_loc = np.concatenate((series_comf_cases_test, comf_cases_pred), axis=0)

        fatal_loc = np.concatenate((series_fatal_test, fatal_pred), axis=0)



        predictions.loc[predictions['Province_State'] == state, 'ConfirmedCases'] = conf_cases_loc.astype(int)

        predictions.loc[predictions['Province_State'] == state, 'Fatalities'] = fatal_loc.astype(int)

        

    return predictions
prediction_final = fill_excl_pred(predictions, training_data_original, test_data)
prediction_final[prediction_final['Country_Region'] == 'Italy']
predictions_csv = pd.DataFrame()



predictions_csv['ForecastId'] = prediction_final['ForecastId'].astype(int)

predictions_csv['ConfirmedCases'] = prediction_final['ConfirmedCases'].astype(float)

predictions_csv['Fatalities'] = prediction_final['Fatalities'].astype(float)
predictions_csv.to_csv('submission.csv', index = False)