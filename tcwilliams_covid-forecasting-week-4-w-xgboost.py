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
import datetime

import pdb

import xgboost as xgb

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, KFold

import matplotlib.pyplot as plt
verbose = False

loop_logic = True    # loop on country/state in main program

scale_data = False   # scale data with MinMaxScaler

use_base_model = True  # use base model throughout

one_hot_encode = False  # use one-hot encoding or label to integer encoding

estimators=5000

# only submit predictions up to last date in test set during public leaderboard period

public_leaderboard_end_date = None # run for final submission
def fix_data_issues(df):

    df['Province_State'] = np.where(df['Province_State'].isnull(), df['Country_Region'], df['Province_State']) 
def get_test_train_for_country_state(one_hot_encode_flag, df_train, df_test, country, state):

    if one_hot_encode_flag == True:

        cs_train = df_train[(df_train['Country_Region_'+country] == 1) & (df_train['Province_State_'+state] == 1) ]

        cs_test = df_test[(df_test['Country_Region_'+country] == 1) & (df_test['Province_State_'+state] == 1) ]

    else:

        cs_train = df_train[(df_train['Country_Region'] == country) & (df_train['Province_State'] == state) ]

        cs_test = df_test[(df_test['Country_Region'] == country) & (df_test['Province_State'] == state) ]

    

    return (cs_train, cs_test)
def transform_dates(df):

    dates = pd.to_datetime(df['Date']) 

    min_dates = dates.min()

#    df['Date_Days_Since_Pandemic_Start'] = (dates - min_dates).dt.days

    df['Date_Year'] = dates.dt.year

    df['Date_Month'] = dates.dt.month

    df['Date_Day'] = dates.dt.day

#    df['Date_Week'] = dates.dt.week

#    df['Date_DayofWeek'] = dates.dt.dayofweek

#    df['Date_DayofYear'] = dates.dt.dayofyear

#    df['Date_WeekofYear'] = dates.dt.weekofyear

#    df['Date_Quarter'] = dates.dt.quarter

    df.drop(['Date'], axis=1, inplace=True)   # remove the date column, no longer needed
def setup_df_encode_and_dates(df, encode_flag, dummy_cols, target_cols=[]):

    # move country in front of province/state

    enc_df = df.copy()

    enc_df = enc_df[[enc_df.columns[0], enc_df.columns[2], enc_df.columns[1],enc_df.columns[3]]]  # 1st column named differently in train vs test

    

    if encode_flag == True:

        enc_df = pd.get_dummies(enc_df, columns=dummy_cols)  # one-hot encoding

    else:

        le = LabelEncoder()

        for dum_col in dummy_cols:

            enc_df[dum_col] = le.fit_transform(enc_df[dum_col])   # label encoding



    # extract date parts / date descriptors (week, quarter, etc.).  Remove original date variable as it can't be used by NN

    transform_dates(enc_df)



    for col in target_cols:

        enc_df[col] = df[col]



    return(enc_df)
def prepare_train_set(df_train):

    # break out main body of train set and separate the target variables out

    train_x, train_target1, train_target2 = df_train.iloc[:, :-2], df_train.iloc[:, -2], df_train.iloc[:, -1]

#    pdb.set_trace()

    return(train_x, train_target1, train_target2)
def prepare_submission(preds):

    preds['ForecastId'] = preds['ForecastId'].fillna(0.0).astype('int32')

    preds['Fatalities'] = preds['Fatalities'].fillna(0.0).astype('int32')

    preds['ConfirmedCases'] = preds['ConfirmedCases'].fillna(0.0).astype('int32')

    preds.clip(lower=0, inplace=True)

    preds.to_csv('submission.csv', index=False)
def model_and_predict(model, X, y, test, estimators=5000):

    if verbose == True:

        print("Initial model ID in model_and_predict: {0}".format(id(model)))

    if model != None:

        run_model = model

        if verbose == True:

            print("Running with model id #{0}".format(id(model)))



    else:

        run_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators= estimators)

        if verbose == True:

            print("Running with new model")



    if verbose == True:

        print("Model ID in model_and_predict: {0}".format(id(run_model)))

    #initial training on 80%/20% train/test split 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)



#    model = model.fit(X_train, y_train)

    run_model.fit(X_train, y_train)



#    y_train_pred = model.predict(X_train)

#    y_test_pred = model.predict(X_test)

    y_train_pred = run_model.predict(X_train)

    y_test_pred = run_model.predict(X_test)

    

#    print("R2: {0:.2f}".format(r2_score(y_train_pred, y_train)))

#    pdb.set_trace()



    # now predict using the trained model on all of the test rows

#    y_pred = model.predict(test)  

    y_pred = run_model.predict(test)  



    y_pred[y_pred < 0] = 0

    

    r2 = r2_score(y_train_pred, y_train, multioutput='variance_weighted')



    return(y_pred, r2)
def show_results(model):

    # Code based on "Selecting Optimal Parameters for XGBoost Model Training" by Andrej Baranovskij (Medium)

    results = model.evals_result()

    epochs = len(results['validation_0']['error'])

    x_axis = range(0, epochs)

    # plot log loss

    fig, ax = plt.subplots()

    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')

    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')

    ax.legend()

    plt.ylabel('Log Loss')

    plt.title('XGBoost Log Loss')

    plt.show()

    # plot classification error

    fig, ax = pyplot.subplots()

    ax.plot(x_axis, results['validation_0']['error'], label='Train')

    ax.plot(x_axis, results['validation_1']['error'], label='Test')

    ax.legend()

    plt.ylabel('Classification Error')

    plt.title('XGBoost Classification Error')

    plt.show()
def fit_models_and_train(country, state, model, train, test):

    X, y_cases, y_fatal = prepare_train_set(train)

    X = X.drop(['Id'], axis=1)    # remove the Id column from the training set to avoid leakage



    forecast_IDs = test.iloc[:,0]  # save the ForecastId column



    test_no_id = test.iloc[:, 1:]   # use the rest of the test set without the ForecastId column



    

    # apply scaling to train and test set

    if scale_data == True:

        scaler = MinMaxScaler()

        X = scaler.fit_transform(X.values)

        test_no_id = scaler.transform(test_no_id.values)



    y_cases_pred, cases_r2 = model_and_predict(model, X, y_cases, test_no_id)   # prior version: estimators = 10000, trying default of 2000

    if verbose == True:

        print("Country {0}, state {1}: cases R2 score: {2:0.2f}.".format(country, state, cases_r2))



#   pdb.set_trace()

    

#    X_train, X_test, y_train, y_test = train_test_split(X, y_fatal, test_size=0.2, random_state=12345)

    y_fatal_pred, fatal_r2 = model_and_predict(model, X, y_fatal, test_no_id)

    if verbose == True:

        print("Country {0}, state {1}: fatalities R2 score: {2:0.2f}.".format(country, state, fatal_r2))



    preds = pd.DataFrame(forecast_IDs)

    preds['ConfirmedCases'] = y_cases_pred

    preds['Fatalities'] = y_fatal_pred



    return(preds)
def cv_model(country, state, train, test):

    X, y_cases, y_fatal = prepare_train_set(train)

    X = X.drop(['Id'], axis=1)    # remove the Id column from the training set to avoid leakage



#    forecast_IDs = test.iloc[:,0]  # save the ForecastId column



    X_test = test.iloc[:, 1:]   # use the rest of the test set without the ForecastId column



    data_train_cases_matrix = xgb.DMatrix(data=X, label=y_cases)

    data_train_fatal_matrix = xgb.DMatrix(data=X, label=y_fatal)

    

#    scores = cross_val_score(model, X, y_cases,cv=5, scoring='accuracy')

#    print("Country {0}, state {1}: cases mean cross-validation score: {2:0.2f}.".format(country, state, scores.mean()))



    cv_results_cases = xgb.cv(dtrain=data_train_cases_matrix, params=parms, nfold=3, num_boost_round=50,

                   early_stopping_rounds=50,metrics="rmse",as_pandas=True,seed=12345)

    

    print("Cases RMSE: {0:.2f}.".format(cv_results_cases['test-rmse-mean'].tail(1).values[0]))

    

    cv_results_fatal = xgb.cv(dtrain=data_train_fatal_matrix, params=parms, nfold=3, num_boost_round=50,

                   early_stopping_rounds=50,metrics="rmse",as_pandas=True,seed=12345)        



    print("Fatalities RMSE: {0:.2f}.".format(cv_results_fatal['test-rmse-mean'].tail(1).values[0]))



#    scores = cross_val_score(model, X, y_fatal,cv=5, scoring='accuracy')

#    print("Country {0}, state {1}: fatalities mean cross-validation score: {2:0.2f}.".format(country, state, scores.mean()))

# Get the training data

df_train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
# Get the test data

df_test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')
# Exploratory data analysis

df_train.shape
df_test.shape
# Fix any known data issues in train and test sets

fix_data_issues(df_train)

fix_data_issues(df_test)
# Save original train/test data in case we need it later

df_train_original = df_train.copy()

df_test_original = df_test.copy()

df_train_original['Datetime'] = pd.to_datetime(df_train_original['Date'])

df_test_original['Datetime'] = pd.to_datetime(df_test_original['Date'])
# remove overlap dates from train set

date_filter = df_train[df_train.Date > "4/1/2020"].index

if not (public_leaderboard_end_date is None):

    df_train.drop(date_filter, inplace=True)  # remove for final submissions
df_train[df_train.Date > '2020/04/01']
df_train_original.head()
parms = {'objective' :'reg:squarederror', 'colsample_bytree' : 0.4, 'learning_rate' : 0.01,

                'max_depth' : 5, 'reg_alpha' : 0.3, 'n_estimators' : 2000 }

num_round = 2
# define base XGBoost parameters and model for predictions

#base_model = xgb.XGBRegressor(objective='reg:squarederror', 

#                         colsample_bytree=0.4, 

#                         learning_rate=0.01,

#                         max_depth=15, 

#                         reg_alpha=0.3,

#                         n_estimators= estimators)



base_model = xgb.XGBRegressor(n_estimators=estimators, random_state=12345, max_depth=15)
print("Model ID: {0}".format(id(base_model)))
# Logic influenced by Anshul Sharma's "COVID19-Explained through Visualizations" notebook,

# RanjitKS's "20 lines; XGBoost; No Leaks; Best Score" and others:



# Set up one-hot encoding to avoid possible leakage from LabelEncoder values (alphabetical ordering of geographies, etc.)



# Possible improvements:

#  - try time lags and other time-series adjustments

#  - try geog, political, transportation, cultural data to enhance model fit



# get country / state list. If one-hot encoded, Train dataframe will have one column per country/state combination

# If label encoded, original columns will have a numeric value instead of text country/state name



if one_hot_encode == True:

    country_groups = df_train_original.groupby(['Country_Region', 'Province_State']).groups

    df_country_list = pd.DataFrame.from_dict(list(country_groups))

    train_country_list = df_country_list[0].unique()



#pdb.set_trace()



df_train_dd = setup_df_encode_and_dates(df_train, one_hot_encode, ['Country_Region', 'Province_State'], ['ConfirmedCases', 'Fatalities'])

df_test_dd = setup_df_encode_and_dates(df_test, one_hot_encode, ['Country_Region', 'Province_State'])





if one_hot_encode == False:

    country_groups = df_train_dd.groupby(['Country_Region', 'Province_State']).groups

    df_country_list = pd.DataFrame.from_dict(list(country_groups))

    train_country_list = df_country_list[0].unique()



#pdb.set_trace()



df_preds = pd.DataFrame({'ForecastId':[],  'ConfirmedCases': [],  'Fatalities': []})



if (loop_logic == True): 

    # loop over states within countries

    print("Starting forecasting for {0} countries.".format(len(train_country_list)))

    for country in train_country_list:

        print("Starting country {0}.".format(country))



        # Get list of states/provinces (if any) for the current country 

        country_states = df_country_list[(df_country_list[0] == country)][1].values



        for state in country_states:

    #        pdb.set_trace()

            # get train / test data for current state/province

            curr_cs_train, curr_cs_test = get_test_train_for_country_state(one_hot_encode, df_train_dd, df_test_dd, country, state)



            # train model for each state/province combination

            # predict state's values (if country values not broken out by state/province, state == country)

            preds = fit_models_and_train(country, state, base_model if use_base_model==True else None, curr_cs_train, curr_cs_test)



            preds = preds.round(5)  # round predictions to 5 decimal places



            # add results to list of predictions

            df_preds = pd.concat([df_preds, preds], axis=0)



    #        show_results(base_model)

        print("Country {0} complete.".format(country))

else:

    print("Starting forecasting for all {0} countries.".format(len(train_country_list)))

    preds = fit_models_and_train("All", "All", base_model if use_base_model==True else None, df_train_dd, df_test_dd)

    df_preds = pd.concat([df_preds, preds], axis=0)

print("All countries complete.")



if not (public_leaderboard_end_date is None):

    # Set predictions to 1 beyond public leaderboard cut-off date if still in pu

    df_preds.loc[(df_test_original.Datetime > pd.to_datetime(public_leaderboard_end_date)), 'ConfirmedCases'] = 1

    df_preds.loc[(df_test_original.Datetime > pd.to_datetime(public_leaderboard_end_date)), 'Fatalities'] = 1

    df_preds[(df_test_original.Datetime > pd.to_datetime(public_leaderboard_end_date))].head()
print(not (public_leaderboard_end_date is None))
df_test_dd.shape
df_preds
df_test_dd.head()
df_train_dd.head()
#show_results(base_model)
if loop_logic == False:

    xgb.plot_importance(base_model)

    plt.rcParams['figure.figsize'] = [40,40]

    plt.show()
df_preds.head()
prepare_submission(df_preds)