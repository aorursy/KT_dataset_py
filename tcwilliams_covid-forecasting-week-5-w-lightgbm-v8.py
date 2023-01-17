# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#from google.colab import drive

#drive.mount('/content/gdrive')

#os.chdir('/content/gdrive/My Drive/kaggle/covid-19/forecasting/week5')

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import datetime

from enum import Enum

import pdb

import lightgbm as lgb

from lightgbm import LGBMRegressor

#import xgboost as xgb

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score, KFold

import matplotlib.pyplot as plt
class ModelType(Enum):

    XGBoost = 1

    LightGBM = 2



model_type = ModelType.LightGBM



verbose = True

loop_logic = True    # loop on country/state/county in main program

scale_data = False   # scale data with MinMaxScaler

use_base_model = False  # use base model throughout

one_hot_encode = False  # use one-hot encoding or label to integer encoding

confidence_levels = [0.05, 0.5, 0.95]

all_lags = [7, 14, 21]

estimators=2500

global_label_encoder = LabelEncoder()



label_encoders = {}



# only submit predictions up to last date in test set during public leaderboard period

public_leaderboard_end_date = None # '5/10/2020' 
def fix_data_issues(df):

    df['Province_State'] = np.where(df['Province_State'].isnull(), df['Country_Region'], df['Province_State']) 

    df['County'] = np.where(df['County'].isnull(), df['Province_State'], df['County']) 
def get_test_train_for_country_state_county(one_hot_encode_flag, df_train, df_test, country, state, county):

    if one_hot_encode_flag == True:

        cs_train = df_train[(df_train['Country_Region_'+country] == 1) & (df_train['Province_State_'+state] == 1)  & (df_train['County_'+county] == 1) ]  

        cs_test = df_test[(df_test['Country_Region_'+country] == 1) & (df_test['Province_State_'+state] == 1) & (df_test['County_'+county] == 1) ]

    else:

        cs_train = df_train[(df_train['Country_Region'] == country) & (df_train['Province_State'] == state)  & (df_train['County'] == county) ]

        cs_test = df_test[(df_test['Country_Region'] == country) & (df_test['Province_State'] == state) & (df_test['County'] == county) ] 

    

    return (cs_train, cs_test)
def add_features(df, default_value=None):

    # add log of cases / fatalities values

    if default_value == None:

        log_cases = np.log1p(df.loc[(df['Target']=='ConfirmedCases'), 'TargetValue'])

        log_fatal = np.log1p(df.loc[(df['Target']=='Fatalities'), 'TargetValue'])

        df.insert(4, 'LogCases', log_cases)

        df.insert(5, 'LogFatal', log_fatal)

        df.insert(5, 'LogPopulation', np.log1p(df['Population']) )

        df['LogCases'].fillna(0)

        df['LogFatal'].fillna(0)

        df['LogPopulation'].fillna(0)

    else:

        df.insert(4, 'LogCases', default_value)

        df.insert(5, 'LogFatal', default_value)

        df.insert(5, 'LogPopulation', default_value) 

        

        

    

    #lag_target(df, all_lags, ['ConfirmedCases', 'Fatalities'], 6)



    #df_group = df.groupby(['Country_Region', 'Province_State'])
def lag_target(df, lags, target_labels, insert_after, all_zeros=False):

    # dataframe dummy columns should still have original names at this point, no one-hot encoding yet    

    country_groups = df.groupby(['Country_Region', 'Province_State', 'County']).groups

    df_country_list = pd.DataFrame.from_dict(list(country_groups))

    unique_country_list = df_country_list[0].unique()

    

    lag_columns = {}

    

#    pdb.set_trace()

    

    for label in target_labels:

        for lag in lags:

            lag_columns[lag] = "{0}_Lag_{1:.0f}".format(label, lag)

            if not lag_columns[lag] in df.columns:

                df.insert(insert_after, lag_columns[lag], 0)



    for country in unique_country_list:

        country_states = df_country_list[(df_country_list[0] == country)][1].values

        for state in country_states:

            state_counties = df_country_list[(df_country_list[0] == country) & (df_country_list[1] == state)][2].values

            for county in state_counties:

                for label in target_labels:

                    geog_filter = (df_train['Country_Region']==country) & (df_train['Province_State']==state) & (df_train['County']==county) & (df_train['Target']==label)

#                    print("Generating lags for {0}, country={1}, state={2}, county={3}, target={4}.".format(lag_columns[lag], country, state, county, label))

                    for lag in lags:

                        if all_zeros == False:

                            df_train.loc[geog_filter, lag_columns[lag]] = df_train.loc[geog_filter, "TargetValue"].shift(lag)

                        else:

                            df_train.loc[geog_filter, lag_columns[lag]] = 0
def transform_dates(df):

    dates = pd.to_datetime(df['Date']) 

    min_dates = dates.min()

#    df['Date_Days_Since_Pandemic_Start'] = (dates - min_dates).dt.days

    df.insert(len(df.columns)-2,'Date_Year', dates.dt.year)

    df.insert(len(df.columns)-2,'Date_Month', dates.dt.month)

    df.insert(len(df.columns)-2,'Date_Day', dates.dt.day)

    df.insert(len(df.columns)-2,'Date_Week', dates.dt.week)

    df.insert(len(df.columns)-2,'Date_DayofWeek', dates.dt.dayofweek)

    df.insert(len(df.columns)-2,'Date_DayofYear', dates.dt.dayofyear)

    df.insert(len(df.columns)-2,'Date_WeekofYear', dates.dt.weekofyear)

    df.insert(len(df.columns)-2,'Date_Quarter', dates.dt.quarter)

    df.drop(['Date'], axis=1, inplace=True)   # remove the date column, no longer needed
def setup_df_encode_and_dates(df, encode_flag, dummy_cols, target_cols=[]):

    # move country in front of province/state

    enc_df = df.copy()

    

    # Find out how to move columns - only the 4 columns listed below are moved



    cols = list(enc_df.columns)

    a, b = cols.index('Province_State'), cols.index('Country_Region')

    cols[b], cols[a] = cols[a], cols[b]

    enc_df = enc_df[cols]

    

#    enc_df = enc_df[[enc_df.columns[0], enc_df.columns[2], enc_df.columns[1],enc_df.columns[3]]]  # 1st column named differently in train vs test

    

    if encode_flag == True:

        enc_df = pd.get_dummies(enc_df, columns=dummy_cols)  # one-hot encoding

#        dummy_df = pd.get_dummies(enc_df, columns=dummy_cols)  # one-hot encoding

#        enc_df = pd.concat([enc_df, dummy_df], axis=1)

#        enc_df.drop(dummy_cols, axis=1)

        enc_df = enc_df[[col for col in enc_df if col not in target_cols] + target_cols]

    else:

        for dum_col in dummy_cols:

            label_encoders[dum_col] = LabelEncoder()

            enc_df[dum_col] = label_encoders[dum_col].fit_transform(enc_df[dum_col])   # label encoding



    # extract date parts / date descriptors (week, quarter, etc.).  Remove original date variable as it can't be used by NN

    transform_dates(enc_df)



    # make sure added feature columns are moved to encoded df

    

    

#    for col in target_cols:

#        enc_df[col] = df[col]



    return(enc_df)
def prepare_train_set(df_train):

    # break out main body of train set and separate the target variables out

    df_cases = df_train[df_train['Target'] == 'ConfirmedCases']

    df_fatal = df_train[df_train['Target'] == 'Fatalities']

    

    train_x_cases, train_target_cases = df_cases.iloc[:, :-2], df_cases.iloc[:, -1]

    train_x_fatal, train_target_fatal = df_fatal.iloc[:, :-2], df_fatal.iloc[:, -1]

    

    return(train_x_cases, train_x_fatal, train_target_cases, train_target_fatal)
def model_and_predict(model, X, y, test, conf_levels, estimators=5000):

    if verbose == True:

        print("Initial model ID in model_and_predict: {0}".format(id(model)))

    if model != None:

        run_model = model

        if verbose == True:

            print("Running with model id #{0}".format(id(model)))



    else:

#        run_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators= estimators)

        if model_type == ModelType.XGBoost:

            run_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators= estimators)

        else:

            #[LightGBM] [Warning] Unknown parameter: loss

            #[LightGBM] [Warning] min_data_in_leaf is set=500, min_child_samples=2 will be ignored. Current value: min_data_in_leaf=500

            #[LightGBM] [Warning] Accuracy may be bad since you didn't set num_leaves and 2^max_depth > num_leaves



            run_model = LGBMRegressor(num_leaves = 85, learning_rate =10**-1.89,n_estimators=100,

                                      min_sum_hessian_in_leaf=(10**-4.1),min_child_samples =2,

                                      subsample =0.97,subsample_freq=10,

                                      colsample_bytree = 0.68,reg_lambda=10**1.4,random_state=1234,n_jobs=4)

        if verbose == True:

            print("Running with new model")



    if verbose == True:

        print("Model ID in model_and_predict: {0}".format(id(run_model)))

        

    #initial training on 80%/20% train/test split 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)



    X_train_weights = X_train['Weight']

    X_test_weights = X_test['Weight']



    X_no_weights = X.drop(['Weight'], axis=1)  # drop Weight column from X for final training

    test_no_weights = test.drop(['Weight'], axis=1)  # drop Weight column from test for final training

    

    # drop weight columns from X and test 

    X_train.drop(['Weight'], axis=1, inplace=True)  # drop Weight column from train set

    X_test.drop(['Weight'], axis=1, inplace=True)  # drop Weight column from test set



    

#    model = model.fit(X_train, y_train)

    if model_type == ModelType.XGBoost:

        run_model.fit(X_train, y_train)

        y_train_pred = run_model.predict(X_train)

        y_test_pred = run_model.predict(X_test)

    else:

#        pdb.set_trace()

#        run_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=5)

#        run_model.fit(X_train, y_train)

        lgb_train = lgb.Dataset(X_train, y_train)

        quantile_models = {}

        y_preds = []

        y_train_preds = []

        lgb_parms = run_model.get_params()

        lgb_parms.pop('silent')

        lgb_parms.pop('importance_type')

        lgb_parms.pop('min_child_weight')        

        for cl in conf_levels:

            lgb_parms['verbose'] = -1

            lgb_parms['alpha'] = cl

            curr_model = lgb.train(lgb_parms, lgb_train)

            quantile_models[cl] = curr_model

            

            y_train_pred = curr_model.predict(X_train)

            y_train_preds.append(y_train_pred)

            

            y_test_pred = curr_model.predict(X_test)

            

            # now predict using the trained model on all of the test rows

            

#            lgb_full = lgb.Dataset(X_no_weights, y)

#            full_model = lgb.train(lgb_parms, lgb_full)

            y_train_full = curr_model.predict(X_no_weights)    # predict with full train set w/o Weight column

            y_pred = curr_model.predict(test_no_weights)

#            y_pred[y_pred < 0] = 0

            y_preds.append(y_pred)

            

    # compute pinball loss here.  

    pb_loss = weighted_pinball_loss(y_train, y_train_preds, X_train_weights, conf_levels)



    

    return(y_preds, pb_loss)
def compute_cl_loss(y, y_hat, weights, tau):

    return((weights * (tau * (y - y_hat) + (1 - tau) * (y_hat - y))))
def weighted_pinball_loss(y, y_hat_arrays, w, tau_list):

    Nf = len(y)

    Nt = len(tau_list)

    

#    w_conf = 1/(np.ln(y.size + 1))

#    w_fatal = 1/(10*np.ln(y.size + 1))



    



    score = (1/Nf) * np.sum([np.sum((1/Nt) * (w * np.maximum(tau * (y - y_hat), (1 - tau) * (y_hat - y)))) for y_hat, tau in zip(y_hat_arrays, tau_list)  ])

    

    return(score)
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
def fit_models_and_train(country, state, county, model, train, test, conf_levels):

    X_cases, X_fatal, y_cases, y_fatal = prepare_train_set(train)

    

    X_cases.drop(['Id'], axis=1, inplace=True)    # remove the Id column from the training set to avoid leakage

    X_fatal.drop(['Id'], axis=1, inplace=True)    # remove the Id column from the training set to avoid leakage



    X_cases_weights = X_cases['Weight']

    X_fatal_weights = X_fatal['Weight']

    

    forecast_IDs = test.iloc[:,0]  # save the ForecastId column

    

    test_cases = test[test['Target'] == 'ConfirmedCases']

    test_fatal = test[test['Target'] == 'Fatalities']



    cases_forecast_IDs = test_cases.ForecastId  # save the ForecastId column for cases

    fatal_forecast_IDs = test_fatal.ForecastId  # save the ForecastId column for fatalities





    test_cases_no_id = test_cases.iloc[:, 1:]   # use the rest of the test set without the ForecastId column

    test_fatal_no_id = test_fatal.iloc[:, 1:]   # use the rest of the test set without the ForecastId column



    test_cases_no_id.drop(['Target'], axis=1, inplace=True)  # drop Target column from test set

    test_fatal_no_id.drop(['Target'], axis=1, inplace=True)  # drop Target column from test set



    country_text = decode_label('Country_Region', country)

    state_text = decode_label('Province_State', state)

    county_text = decode_label('County', county)

    

    # apply scaling to train and test set

    if scale_data == True:

        scaler = MinMaxScaler()

        X = scaler.fit_transform(X.values)

        test_cases_no_id = scaler.transform(test_cases_no_id.values)

        test_fatal_no_id = scaler.transform(test_fatal_no_id.values)



#    pdb.set_trace()



    if verbose == True:

        print("Predicting cases.")

        

    y_cases_pred, cases_pb_loss = model_and_predict(model, X_cases, y_cases, test_cases_no_id, conf_levels)   # prior version: estimators = 10000, trying default of 2000

    if verbose == True:

        print("Country {0}, state {1}, county {2}: cases pinball loss score: {3:0.2f}.".format(country_text, state_text, county_text, cases_pb_loss))



#   pdb.set_trace()

    

##    X_train, X_test, y_train, y_test = train_test_split(X, y_fatal, test_size=0.2, random_state=12345)

    if verbose == True:

        print("Predicting fatalities.")

        

    y_fatal_pred, fatal_pb_loss = model_and_predict(model, X_fatal, y_fatal, test_fatal_no_id, conf_levels)

    if verbose == True:

        print("Country {0}, state {1}, county {2}: fatalities pinball loss score: {3:0.2f}.".format(country_text, state_text, county_text, fatal_pb_loss))



#    5/9:  Make sure preds are in proper format for submission file



    preds = pd.DataFrame(forecast_IDs)



# Stitch the predictions back together



#    preds['ConfirmedCases'] = y_cases_pred

#    preds['Fatalities'] = y_fatal_pred





#    pdb.set_trace()



    y_cases_with_index = []

    for i in range(len(cases_forecast_IDs)):

        conf_group = []

        for j in range(len(conf_levels)):

            conf_group.append([ cases_forecast_IDs[cases_forecast_IDs.index[i]], conf_levels[j], y_cases_pred[j][i]])

        y_cases_with_index.append(conf_group)



    y_fatal_with_index = []

    for i in range(len(fatal_forecast_IDs)):

        conf_group = []

        for j in range(len(conf_levels)):

            conf_group.append([ fatal_forecast_IDs[fatal_forecast_IDs.index[i]], conf_levels[j], y_fatal_pred[j][i]])

        y_fatal_with_index.append(conf_group)





    preds = [ y_cases_with_index, y_fatal_with_index ]



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

def decode_label(column, label):

    decoded_label = label     # return original label if no decoding to be done

    

    if one_hot_encode == False:

        if label != "All":

            decoded_label = label_encoders[column].inverse_transform([label])

        

    return(decoded_label)
def prepare_submission(df_all_preds):

    formatted_preds = [ ["{0:.0f}_{1}".format(row[0], row[1] ), row[2]] for row in df_all_preds[['ForecastId', 'Quantile', 'TargetValue']].values ]

    pd.DataFrame(formatted_preds).to_csv('submission.csv', header=['ForecastId_Quantile', 'TargetValue'], index=False)
# Get the training data

df_train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')
# Get the test data

df_test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')
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

if not (public_leaderboard_end_date == None):

    date_filter = df_train[pd.to_datetime(df_train.Date) > pd.to_datetime(public_leaderboard_end_date)].index

    df_train.drop(date_filter, inplace=True)  # remove for final submissions
df_train[pd.to_datetime(df_train.Date) > pd.to_datetime('2020/04/01')]
df_train_original.head()
# add features

#add_features(df_train)

#add_features(df_test, 0)
df_train.head(92)
df_test.head(92)
if model_type == ModelType.XGBoost:

    parms = {'loss': 'quantile', 'objective' :'reg:squarederror', 'colsample_bytree' : 0.4, 'learning_rate' : 0.01,

                    'max_depth' : 5, 'reg_alpha' : 0.3, 'n_estimators' : 2000 }

else:

    if loop_logic == False:

        parms = {'verbose' : - 1, 'objective' :'quantile',  'max_depth' : 8, 'num_leaves' : 50, 'colsample_bytree' : 0.4, 'learning_rate' : 10**-1.89, 

                         'reg_alpha' : 0.3, 'n_estimators' : 1000, 'min_sum_hessian_in_leaf' : (10**-4.1), 

                         'min_child_samples' : 2, 'subsample' : 0.97, 'subsample_freq' : 10, 'min_data_in_leaf' : 500, 

                         'colsample_bytree' : 0.68, 'reg_lambda' : 10**1.4, 'random_state' : 1234, 'n_jobs': 4 

                }

    else:

        parms = {'verbose' : - 1, 'objective' :'quantile',  'max_depth' : 8, 'num_leaves' : 50, 'colsample_bytree' : 0.4, 'learning_rate' : 10**-1.89, 

                         'reg_alpha' : 0.3, 'n_estimators' : 1000, 'min_sum_hessian_in_leaf' : (10**-4.1), 

                         'min_child_samples' : 2, 'subsample' : 0.97, 'subsample_freq' : 10, 

                         'colsample_bytree' : 0.68, 'reg_lambda' : 10**1.4, 'random_state' : 1234, 'n_jobs': 4 

                }

   

num_round = 2
# define base XGBoost parameters and model for predictions

#base_model = xgb.XGBRegressor(objective='reg:squarederror', 

#                         colsample_bytree=0.4, 

#                         learning_rate=0.01,

#                         max_depth=15, 

#                         reg_alpha=0.3,

#                         n_estimators= estimators)



if model_type == ModelType.XGBoost:

    base_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=estimators, random_state=12345, max_depth=15)

#    base_model = xgb.XGBRegressor(objective='reg:squarederror', 

#                             colsample_bytree=0.4, 

#                             learning_rate=0.01,

#                             max_depth=15, 

#                             reg_alpha=0.3,

#                             n_estimators= estimators)

else:

    base_model = LGBMRegressor(objective='quantile', num_leaves = 85, learning_rate =10**-1.89,n_estimators=100,

                               min_sum_hessian_in_leaf=(10**-4.1),min_child_samples =2,subsample =0.97,subsample_freq=10,

                               colsample_bytree = 0.68,reg_lambda=10**1.4,random_state=1234,n_jobs=4, verbose=-1)
print("Model ID: {0}".format(id(base_model)))
# Logic influenced by Anshul Sharma's "COVID19-Explained through Visualizations" notebook,

# RanjitKS's "20 lines; XGBoost; No Leaks; Best Score" and others:



# Set up one-hot encoding to avoid possible leakage from LabelEncoder values (alphabetical ordering of geographies, etc.)



# Possible improvements:

#  - try time lags and other time-series adjustments

#  - try geog, political, transportation, cultural data to enhance model fit



# get country / state list. If one-hot encoded, Train dataframe will have one column per country/state combination

# If label encoded, original columns will have a numeric value instead of text country/state name



#pdb.set_trace()



df_train_dd = setup_df_encode_and_dates(df_train, one_hot_encode, ['Country_Region', 'Province_State', 'County'], ['Target', 'TargetValue'])

df_test_dd = setup_df_encode_and_dates(df_test, one_hot_encode, ['Country_Region', 'Province_State', 'County'])



if one_hot_encode == True:

    country_groups = df_train_original.groupby(['Country_Region', 'Province_State', 'County']).groups

else:

    country_groups = df_train_dd.groupby(['Country_Region', 'Province_State', 'County']).groups

    

df_country_list = pd.DataFrame.from_dict(list(country_groups))

train_country_list = df_country_list[0].unique()



#pdb.set_trace()



#df_preds = pd.DataFrame({'TargetValue': []})





if (loop_logic == True): 

    preds = [[],[]]

    # loop over states within countries

    print("Starting forecasting for {0} countries.".format(len(train_country_list)))

    for country in train_country_list:

        country_text = decode_label('Country_Region', country)

        print("Starting country {0}.".format(country_text))



        # Get list of states/provinces (if any) for the current country 

        country_states = np.unique(df_country_list[(df_country_list[0] == country)][1].values)

        

        for state in country_states:

            state_text = decode_label('Province_State', state)

            print("Starting state {0}.".format(state_text))

            # Get list of counties (if any) in current state

            state_counties = np.unique(df_country_list[(df_country_list[0] == country) & (df_country_list[1] == state)][2].values)

            for county in state_counties:

        #        pdb.set_trace()

                county_text = decode_label('County', county)

                print("Starting county {0}.".format(county_text))

                # get train / test data for current state/province/county

                curr_cs_train, curr_cs_test = get_test_train_for_country_state_county(one_hot_encode, df_train_dd, df_test_dd, country, state, county)



                # train model for each state/province/county combination

                # predict county's values (if country values not broken out by county/state/province, county == state == country)

                curr_preds = fit_models_and_train(country, state, county, base_model if use_base_model==True else None, curr_cs_train, curr_cs_test, confidence_levels)



#                preds = [ np.round(pred_array, 5) for pred_array in preds ]  # round predictions to 5 decimal places

                

#                pdb.set_trace()

        

#                cases_test_ids = curr_cs_test[curr_cs_test['Target']=='ConfirmedCases']['ForecastId']

#                fatal_test_ids = curr_cs_test[curr_cs_test['Target']=='Fatalities']['ForecastId']

#                cases_pred_dict = { cases_test_ids : preds[0]}

#                fatal_pred_dics = { fatal_test_ids : preds[1]}



#                pdb.set_trace()



                for i in range(0, len(curr_preds[0])):

                    preds[0].append(curr_preds[0][i])

                    preds[1].append(curr_preds[1][i])

                

#                # add results to list of predictions

#                df_preds = pd.concat([df_preds, cases_pred_dict], axis=0)

#                df_preds = pd.concat([df_preds, fatal_pred_dict], axis=0)



    #        show_results(base_model)

        print("Country {0} complete.".format(country))

else:

    print("Starting forecasting for all {0} countries.".format(len(train_country_list)))

    preds = fit_models_and_train("All", "All", "All", base_model if use_base_model==True else None, df_train_dd, df_test_dd, confidence_levels)

#    df_preds = pd.concat([df_preds, preds], axis=0)

    

print("All countries complete.")
df_train.head()
all_preds = preds[0][0] + preds[1][0]    #  cases and fatalities

for i in range(1, len(preds[0])):         #### check if range 1 is correct start #####

    all_preds += preds[0][i] + preds[1][i]

#    preds = preds.fillna(0.0).astype('int32')

  
df_all_preds = pd.DataFrame(all_preds, columns=['ForecastId','Quantile', 'TargetValue'])
df_all_preds['TargetValue'].clip(lower=0, inplace=True)
df_all_preds.head()
if not public_leaderboard_end_date is None:

    date_cutoff_forecast_ids = df_test[(df_test_original.Datetime > pd.to_datetime(public_leaderboard_end_date))].ForecastId

    df_all_preds.loc[df_all_preds['ForecastId'].isin(date_cutoff_forecast_ids), 'TargetValue'] = 1
df_all_preds[df_all_preds['ForecastId'] == 31]
if loop_logic == False:

    if model_type == ModelType.XGBoost:

        xgb.plot_importance(base_model)

        plt.rcParams['figure.figsize'] = [40,40]

        plt.show()
prepare_submission(df_all_preds)