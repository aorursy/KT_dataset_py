# Loading submission file generated below as time was up

# and I had not enough time to run this before competition

# ended

import pandas as pd

submission = pd.read_csv('/kaggle/input/resultscov19week2/submission (2).csv')

submission
submission.to_csv('submission.csv', index=False)



# Everything below this generates the submission file above.
# # Imports

# import pandas as pd

# import numpy as np

# import matplotlib.pyplot as plt

# %matplotlib inline

# plt.style.use('fivethirtyeight')

# import warnings

# warnings.filterwarnings("ignore")

# from tensorflow.keras.models import Sequential

# from tensorflow.keras.layers import Dense, LSTM

# from tensorflow.keras.callbacks import EarlyStopping

# from lightgbm import LGBMRegressor

# import time

# from sklearn.model_selection import cross_val_score
# # Loading data

# df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

# df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

# submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')

# df_train.head()
# # Making Date become timestamp

# from datetime import datetime

# df_train['Date'] = pd.to_datetime(df_train['Date'])

# df_test['Date'] = pd.to_datetime(df_test['Date'])



# df_train['Date'] = df_train['Date'].apply(lambda s: time.mktime(s.timetuple()))

# df_test['Date'] = df_test['Date'].apply(lambda s: time.mktime(s.timetuple()))



# min_timestamp = np.min(df_train['Date'])

# df_train['Date'] = df_train['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)

# df_test['Date'] = df_test['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)

# df_test.head()
# df_intersection = df_test[df_test['Date'] <= np.max(df_train['Date'])]

# df_intersection
# # Following the idea at

# # https://www.kaggle.com/ranjithks/25-lines-of-code-results-better-score#Fill-NaN-from-State-feature

# # Filling NaN states with the Country



# EMPTY_VAL = "EMPTY_VAL"



# def fillState(state, country):

#     if state == EMPTY_VAL: return country

#     return state



# def replaceGeorgiaState (state, country):

#     if (state == 'Georgia') and (country == 'US'):

#         return 'Georgia_State'

#     else:

#         return state



# df_train['Province_State'].fillna(EMPTY_VAL, inplace=True)

# df_train['Province_State'] = df_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

# df_train['Province_State'] = df_train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : replaceGeorgiaState(x['Province_State'], x['Country_Region']), axis=1)



# df_test['Province_State'].fillna(EMPTY_VAL, inplace=True)

# df_test['Province_State'] = df_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

# df_test['Province_State'] = df_test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : replaceGeorgiaState(x['Province_State'], x['Country_Region']), axis=1)



# df_intersection['Province_State'].fillna(EMPTY_VAL, inplace=True)

# df_intersection['Province_State'] = df_intersection.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

# df_intersection['Province_State'] = df_intersection.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : replaceGeorgiaState(x['Province_State'], x['Country_Region']), axis=1)



# df_train[df_train['Province_State'] == 'Georgia_State']
# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()

# df_train['country_code'] = le.fit_transform(df_train['Country_Region'])

# df_test ['country_code'] = le.transform(df_test['Country_Region'])

# df_intersection ['country_code'] = le.transform(df_intersection['Country_Region'])



# le = LabelEncoder()

# df_train['province_code'] = le.fit_transform(df_train['Province_State'])

# df_test ['province_code'] = le.transform(df_test['Province_State'])

# df_intersection ['province_code'] = le.transform(df_intersection['Province_State'])



# df_train[df_train['Province_State'] == 'Georgia_State']
# # Adding validation data into the Intersection DF

# states = sorted(set(df_intersection['Province_State']))

# df_intersection['ConfirmedCases'] = float('NaN')

# df_intersection['Fatalities'] = float('NaN')



# for state in states:

#     dates = sorted(set(df_intersection[df_intersection['Province_State'] == state]['Date']))

#     min_date = np.min(dates)

#     max_date = np.max(dates)

#     idx = df_intersection[df_intersection['Province_State'] == state].index

#     values = df_train[(df_train['Province_State'] == state) & (df_train['Date'] >= min_date) & (df_train['Date'] <= max_date)][['ConfirmedCases', 'Fatalities']].values

#     values = pd.DataFrame(values, index = list(idx), columns=['ConfirmedCases', 'Fatalities'])

#     df_intersection['ConfirmedCases'].loc[idx] = values['ConfirmedCases']

#     df_intersection['Fatalities'].loc[idx] = values['Fatalities']

# df_intersection
# # Check if any Province_State value on test dataset isn't on train dataset

# # If nothing prints, everything is okay

# for a in set(df_test['Province_State']):

#     if a not in set(df_train['Province_State']):

#         print (a)
# from sklearn.pipeline import make_pipeline

# from sklearn.preprocessing import PolynomialFeatures, LabelEncoder

# from sklearn.linear_model import LinearRegression, BayesianRidge, Lasso

# from sklearn.metrics import mean_squared_log_error



# from sklearn.ensemble.weight_boosting import AdaBoostRegressor

# from sklearn.linear_model.base import LinearRegression

# from sklearn.linear_model.passive_aggressive import PassiveAggressiveRegressor

# from sklearn.linear_model.theil_sen import TheilSenRegressor



# def handle_predictions (predictions, lowest = 0):

#     #predictions = np.round(predictions, 0)

#     # Predictions can't be negative

#     predictions[predictions < 0] = 0

#     # Predictions can't decrease from greatest value on train dataset

#     predictions[predictions < lowest] = lowest

#     # Predictions can't decrease over time

#     for i in range(1, len(predictions)):

#         if predictions[i] < predictions[i - 1]:

#             predictions[i] = predictions[i - 1]

#     #return predictions.astype(int)

#     return predictions



# def fillSubmission (state, column, values,):

#     idx = df_test[df_test['Province_State'] == state].index

#     values = pd.DataFrame(np.array(values), index = list(idx), columns=[column])

#     submission[column].loc[idx] = values[column]

#     return submission



# def avg_rmsle():

#     idx = df_intersection.index

#     my_sub = df_test.loc[idx][['ConfirmedCases', 'Fatalities']]

#     cases_pred = my_sub['ConfirmedCases'].values

#     fatal_pred = my_sub['Fatalities'].values

#     cases_targ = df_intersection.loc[idx]['ConfirmedCases'].values

#     fatal_targ = df_intersection.loc[idx]['Fatalities'].values

#     cases_score = np.sqrt(mean_squared_log_error( cases_targ, cases_pred ))

#     fatal_score = np.sqrt(mean_squared_log_error( fatal_targ, fatal_pred ))

#     score = (cases_score + fatal_score)/2

#     return score
# def checkState (state):

#     idx = df_test[df_test['Province_State'] == state].index

#     return df_test.loc[idx]



# def plotStatus (states):

#     if type(states) == list:

#         for state in states:

#             plt.figure(figsize=(14,8))

#             plt.title('COVID-19 cases on {}'.format(states))

#             df = df_train[df_train['Province_State'] == state]

#             test = df_test[df_test['Province_State'] == state]

#             intersection = df_intersection[df_intersection['Province_State'] == state]

#             idx = df_test[df_test['Province_State'] == state].index

#             legend = []

#             plt.xlabel('#Days since dataset')

#             plt.ylabel('Number')

#             plt.plot(df['Date'], df['ConfirmedCases'])

#             plt.plot(test['Date'], test['ConfirmedCases'])

#             #plt.plot(intersection['Date'], intersection['ConfirmedCases'])

#             legend.append('{} confirmed cases'.format(state))

#             legend.append('{} predicted cases'.format(state))

#             #legend.append('{} actual cases'.format(state))

#             plt.legend(legend)

#             plt.show()

#             legend = []

#             plt.figure(figsize=(14,8))

#             plt.title('COVID-19 fatalities on {}'.format(states))

#             plt.xlabel('#Days since dataset')

#             plt.ylabel('Number')

#             plt.plot(df['Date'], df['Fatalities'])

#             plt.plot(test['Date'], test['Fatalities'])

#             #plt.plot(intersection['Date'], intersection['Fatalities'])

#             legend.append('{} fatalities'.format(state))

#             legend.append('{} predicted fatalities'.format(state))

#             #legend.append('{} actual fatalities'.format(state))

#             plt.show()

#     else:

#         state = states

#         df = df_train[df_train['Province_State'] == state]

#         plt.figure(figsize=(14,8))

#         plt.xlabel('#Days since dataset')

#         plt.ylabel('Number')

#         plt.plot(df['Date'], df['ConfirmedCases'])

#         plt.plot(df['Date'], df['Fatalities'])

#         plt.legend(['Confirmed cases', 'Fatalities'])

#     plt.show()



# raw_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

# raw_test  = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

# raw_train['Date'] = pd.to_datetime(raw_train['Date'], infer_datetime_format=True)

# raw_test['Date']  = pd.to_datetime(raw_test['Date'], infer_datetime_format=True)



# def rmsle (state):

#     idx = df_intersection[df_intersection['Province_State'] == state].index

#     my_sub = df_test.loc[idx][['ConfirmedCases', 'Fatalities']]

#     cases_pred = my_sub['ConfirmedCases'].values

#     fatal_pred = my_sub['Fatalities'].values

#     cases_targ = df_intersection.loc[idx]['ConfirmedCases'].values

#     fatal_targ = df_intersection.loc[idx]['Fatalities'].values

#     cases = np.sqrt(mean_squared_log_error( cases_targ, cases_pred ))

#     fatal = np.sqrt(mean_squared_log_error( fatal_targ, fatal_pred ))

#     return cases, fatal
# from pandas.plotting import autocorrelation_plot



# plt.figure(figsize=(14,8))

# autocorrelation_plot(df_train[ df_train['Province_State'] == 'Brazil' ]['ConfirmedCases'])

# plt.show()
# import time

# from tqdm import tqdm



# start_time = time.time()



# lag_range = np.arange(1,8,1)



# with tqdm(total = len(list(states))) as pbar:

#     for state in states:

#         for d in df_train['Date'].drop_duplicates():

#             mask = (df_train['Date'] == d) & (df_train['Province_State'] == state)

#             for lag in lag_range:

#                 mask_org = (df_train['Date'] == (d - lag)) & (df_train['Province_State'] == state)

#                 try:

#                     df_train.loc[mask, 'ConfirmedCases_' + str(lag)] = df_train.loc[mask_org, 'ConfirmedCases'].values

#                 except:

#                     df_train.loc[mask, 'ConfirmedCases_' + str(lag)] = 0

#                 try:

#                     df_train.loc[mask, 'Fatalities_' + str(lag)] = df_train.loc[mask_org, 'Fatalities'].values

#                 except:

#                     df_train.loc[mask, 'Fatalities_' + str(lag)] = 0

#         pbar.update(1)

# print('Time spent for building features is {} minutes'.format(round((time.time()-start_time)/60,1)))
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from tensorflow.keras.layers import Dropout

# import keras.backend as K



# def root_mean_squared_log_error(y_true, y_pred):

#     return K.sqrt(K.mean(K.square(K.log(y_pred + 1) - K.log(y_true + 1)))) 



# cases_columns = ['country_code', 'province_code', 'Date', 'ConfirmedCases_1', 'ConfirmedCases_2', 'ConfirmedCases_3', 'ConfirmedCases_4', 'ConfirmedCases_5', 'ConfirmedCases_6', 'ConfirmedCases_7']

# fatal_columns = set(['country_code', 'province_code', 'Date', 'Fatalities_1', 'Fatalities_2', 'Fatalities_3', 'Fatalities_4', 'Fatalities_5', 'Fatalities_6', 'Fatalities_7'] + cases_columns)



# X_cases_scaler = StandardScaler()

# X_fatal_scaler = StandardScaler()

# # Maybe by scaling y from 0 to something lower than 1,

# # it makes it possible for predicting things greater

# # than the greatest value on y more accurately.

# y_cases_scaler = MinMaxScaler(feature_range=(0, .7)) # The max number of confirmed cases on a single state/province has already reached next to its' greatest

# y_fatal_scaler = MinMaxScaler(feature_range=(0, .3)) # The number of fatalities maybe not



# # Setting patience for training

# es = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=25)



# # Getting datasets

# X_cases = df_train[cases_columns].values

# X_fatal = df_train[fatal_columns].values

# y_cases = df_train['ConfirmedCases'].values.reshape(-1, 1)

# y_fatal = df_train['Fatalities'].values.reshape(-1, 1)



# # Scaling datasets

# X_cases = X_cases_scaler.fit_transform(X_cases)

# X_fatal = X_fatal_scaler.fit_transform(X_fatal)

# y_cases = y_cases_scaler.fit_transform(y_cases)

# y_fatal = y_fatal_scaler.fit_transform(y_fatal)



# # Fixing shapes

# X_cases = X_cases.reshape(X_cases.shape[0], 1, X_cases.shape[1])

# X_fatal = X_fatal.reshape(X_fatal.shape[0], 1, X_fatal.shape[1])



# # # 0.029632697546336188

# # # Average = 0.1760559035529777, 0.21325053819565856

# # # Modeling for cases

# # model_cases = Sequential()

# # model_cases.add(LSTM(60, return_sequences=True, input_shape=(1, len(cases_columns)), activation='softplus'))

# # model_cases.add(LSTM(60, activation='relu'))

# # model_cases.add(Dropout(0.2))

# # model_cases.add(Dense(50, activation='sigmoid'))

# # model_cases.add(Dense(1, activation='sigmoid'))

# # model_cases.compile(loss=root_mean_squared_log_error, optimizer='adam')



# # # Modeling for fatal

# # model_fatal = Sequential()

# # model_fatal.add(LSTM(60, return_sequences=True, input_shape=(1, len(fatal_columns)), activation='softplus'))

# # model_fatal.add(LSTM(60, activation='relu'))

# # model_fatal.add(Dropout(0.2))

# # model_fatal.add(Dense(50, activation='sigmoid'))

# # model_fatal.add(Dense(1, activation='sigmoid'))

# # model_fatal.compile(loss=root_mean_squared_log_error, optimizer='adam')



# # # 0.03529411203400082

# # # Average = 0.1711087062044468, 0.3030005671812312

# # # Modeling for cases

# # model_cases = Sequential()

# # model_cases.add(LSTM(60, return_sequences=True, input_shape=(1, len(cases_columns)), activation='sigmoid'))

# # model_cases.add(Dropout(0.2))

# # model_cases.add(LSTM(60, activation='sigmoid'))

# # model_cases.add(Dropout(0.2))

# # model_cases.add(Dense(50, activation='sigmoid'))

# # model_cases.add(Dense(1, activation='sigmoid'))

# # model_cases.compile(loss=root_mean_squared_log_error, optimizer='adam')



# # # Modeling for fatal

# # model_fatal = Sequential()

# # model_fatal.add(LSTM(60, return_sequences=True, input_shape=(1, len(fatal_columns)), activation='softplus'))

# # model_fatal.add(LSTM(30, activation='relu'))

# # model_fatal.add(Dropout(0.2))

# # model_fatal.add(Dense(10, activation='sigmoid'))

# # model_fatal.add(Dense(1, activation='sigmoid'))

# # model_fatal.compile(loss=root_mean_squared_log_error, optimizer='adam')



# # # 0.03700217774421334

# # # Average = 0.17584334824320144, 0.2999503126896585

# # # Modeling for cases

# # model_cases = Sequential()

# # model_cases.add(LSTM(750, return_sequences=False, input_shape=(1, len(cases_columns)), activation='softplus'))

# # model_cases.add(Dropout(0.2))

# # model_cases.add(Dense(1, activation='sigmoid'))

# # model_cases.compile(loss=root_mean_squared_log_error, optimizer='adam')



# # # Modeling for fatal

# # model_fatal = Sequential()

# # model_fatal.add(LSTM(60, return_sequences=True, input_shape=(1, len(fatal_columns)), activation='relu'))

# # model_fatal.add(Dropout(0.1))

# # model_fatal.add(LSTM(60, activation='relu'))

# # model_fatal.add(Dropout(0.1))

# # model_fatal.add(Dense(10, activation='sigmoid'))

# # model_fatal.add(Dense(1, activation='sigmoid'))

# # model_fatal.compile(loss=root_mean_squared_log_error, optimizer='adam')



# # # 0.027998083663296477

# # # Average = 0.1431641824933455, 0.2134293608876186

# # # Modeling for cases

# # model_cases = Sequential()

# # model_cases.add(LSTM(750, return_sequences=False, input_shape=(1, len(cases_columns)), activation='softplus'))

# # model_cases.add(Dropout(0.2))

# # model_cases.add(Dense(20, activation='sigmoid'))

# # model_cases.add(Dense(1, activation='sigmoid'))

# # model_cases.compile(loss=root_mean_squared_log_error, optimizer='adam')



# # # Modeling for fatal

# # model_cases = Sequential()

# # model_cases.add(LSTM(60, return_sequences=True, input_shape=(1, len(cases_columns)), activation='sigmoid'))

# # model_cases.add(Dropout(0.2))

# # model_cases.add(LSTM(60, activation='sigmoid'))

# # model_cases.add(Dropout(0.2))

# # model_cases.add(Dense(50, activation='sigmoid'))

# # model_cases.add(Dense(1, activation='softplus'))

# # model_cases.compile(loss=root_mean_squared_log_error, optimizer='adam')



# # # 0.07864703262912029

# # # Average = 0.3382936857372058, 0.5043676839090568

# # # Modeling for cases

# # model_cases = Sequential()

# # model_cases.add(LSTM(30, activation='softplus'))

# # model_cases.add(Dropout(0.2))

# # model_cases.add(Dense(15, activation='relu'))

# # model_cases.add(Dense(3, activation='sigmoid'))

# # model_cases.add(Dense(1, activation='sigmoid'))

# # model_cases.compile(loss=root_mean_squared_log_error, optimizer='adam')



# # # Modeling for fatal

# # model_fatal = Sequential()

# # model_fatal.add(LSTM(30, return_sequences=True, input_shape=(1, len(fatal_columns)), activation='softplus'))

# # model_fatal.add(LSTM(20, return_sequences=True, activation='relu'))

# # model_fatal.add(LSTM(10, activation='sigmoid'))

# # model_fatal.add(Dropout(0.2))

# # model_fatal.add(Dense(5, activation='sigmoid'))

# # model_fatal.add(Dense(1, activation='sigmoid'))

# # model_fatal.compile(loss=root_mean_squared_log_error, optimizer='adam')



# # selu seems pretty OK

# # softplus goes low on losses, pessimistic

# # relu goes lower than selu on losses, but TOO SAFE.No rescaling solved. TOO SAFE

# # tanh seems safe too

# # sigmoid performs flawlessly, but on Italy it becomes too safe.

# # linear can go to infinite



# # 0.04121125879913207

# # Average = 0.142672518218051, 0.3815508620346949

# # Modeling for cases

# model_cases = Sequential()

# model_cases.add(LSTM(60, return_sequences=True, input_shape=(1, len(cases_columns)), activation='sigmoid'))

# model_cases.add(Dropout(0.2))

# model_cases.add(LSTM(60, activation='sigmoid'))

# model_cases.add(Dropout(0.2))

# model_cases.add(Dense(50, activation='sigmoid'))

# model_cases.add(Dense(1, activation='softplus'))

# model_cases.compile(loss=root_mean_squared_log_error, optimizer='adam')



# # Modeling for fatal

# model_fatal = Sequential()

# model_fatal.add(LSTM(60, return_sequences=True, input_shape=(1, len(fatal_columns)), activation='sigmoid'))

# model_fatal.add(Dropout(0.2))

# model_fatal.add(LSTM(60, activation='sigmoid'))

# model_fatal.add(Dropout(0.2))

# model_fatal.add(Dense(50, activation='sigmoid'))

# model_fatal.add(Dense(1, activation='softplus'))

# model_fatal.compile(loss=root_mean_squared_log_error, optimizer='adam')
# # Fitting cases

# model_cases.fit(X_cases, y_cases, batch_size=128, epochs=5000, callbacks=[es])
# # Fitting fatal

# model_fatal.fit(X_fatal, y_fatal, batch_size=128, epochs=5000, callbacks=[es])
# input_cols = list(set(cases_columns + list(fatal_columns)))

# output_cols = ['ConfirmedCases', 'Fatalities']

# adj_input_cols = [e for e in input_cols if e not in ('province_code', 'country_code', 'Date')]

# lag_range = np.arange(1,8,1)

# pred_dt_range = range(int(df_test['Date'].min()), int(df_test['Date'].max()) + 1)



# # Making a random set of 10 states in order to validate models

# import random

# random_validation_set = ['Brazil', 'New York', 'Afghanistan', 'Zhejiang', 'Italy']#random.sample(states, 10)

# print ("The random validation set is {}".format(random_validation_set))



# # Filling data for intersection

# for col in (adj_input_cols + output_cols):

#     df_test[col] = float('NaN')

# test_intersection_mask = (df_test['Date'] <= df_train['Date'].max())

# train_intersection_mask = (df_train['Date'] >= df_test['Date'].min())

# df_test.loc[test_intersection_mask, input_cols + output_cols] = df_train.loc[train_intersection_mask, input_cols + output_cols].values
# use_predictions = False



# start_time = time.time()

# with tqdm(total = len(list(states))) as pbar:

#     for state in states:

#         for d in pred_dt_range:

#             mask = (df_test['Date'] == d) & (df_test['Province_State'] == state)

#             if (d > df_train['Date'].max()):

#                 for lag in lag_range:

#                     mask_org = (df_test['Date'] == (d - lag)) & (df_test['Province_State'] == state)

#                     try:

#                         df_test.loc[mask, 'ConfirmedCases_' + str(lag)] = df_test.loc[mask_org, 'ConfirmedCases'].values

#                     except:

#                         df_test.loc[mask, 'ConfirmedCases_' + str(lag)] = 0

#                     try:

#                         df_test.loc[mask, 'Fatalities_' + str(lag)] = df_test.loc[mask_org, 'Fatalities'].values

#                     except:

#                         df_test.loc[mask, 'Fatalities_' + str(lag)] = 0

#             X_test  = df_test.loc[mask, input_cols]

#             # Cases

#             X_test_cases = X_test[cases_columns].values

#             X_test_cases = X_cases_scaler.transform(X_test_cases)

#             X_test_cases.reshape(X_test_cases.shape[0], 1, X_test_cases.shape[1])

#             next_cases = model_cases.predict(np.array([X_test_cases]))

#             next_cases_scaled = y_cases_scaler.inverse_transform(next_cases)

#             # Fatal

#             X_test_fatal = X_test[fatal_columns].values

#             X_test_fatal = X_fatal_scaler.transform(X_test_fatal)

#             X_test_fatal.reshape(X_test_fatal.shape[0], 1, X_test_fatal.shape[1])

#             next_fatal = model_fatal.predict(np.array([X_test_fatal]))

#             next_fatal_scaled = y_fatal_scaler.inverse_transform(next_fatal)

#             # Update df_test

#             if (d > np.max(df_train['Date'].values)):

#                 if (next_cases_scaled[0][0] < 0):

#                     next_cases_scaled[0][0] = 0

#                 if (next_cases_scaled[0][0] < X_test['ConfirmedCases_1'].values[0]):

#                     next_cases_scaled[0][0] = X_test['ConfirmedCases_1'].values[0]

#                 df_test.loc[mask, 'ConfirmedCases'] = next_cases_scaled

#                 if (next_fatal_scaled[0][0] < 0):

#                     next_fatal_scaled[0][0] = 0

#                 if (next_fatal_scaled[0][0] < X_test['Fatalities_1'].values[0]):

#                     next_fatal_scaled[0][0] = X_test['Fatalities_1'].values[0]

#                 df_test.loc[mask, 'Fatalities'] = next_fatal_scaled

#             else:

#                 if use_predictions:

#                     if (next_cases_scaled[0][0] < 0):

#                         next_cases_scaled[0][0] = 0

#                     if (next_cases_scaled[0][0] < X_test['ConfirmedCases_1'].values[0]):

#                         next_cases_scaled[0][0] = X_test['ConfirmedCases_1'].values[0]

#                     df_test.loc[mask, 'ConfirmedCases'] = next_cases_scaled

#                     if (next_fatal_scaled[0][0] < 0):

#                         next_fatal_scaled[0][0] = 0

#                     if (next_fatal_scaled[0][0] < X_test['Fatalities_1'].values[0]):

#                         next_fatal_scaled[0][0] = X_test['Fatalities_1'].values[0]

#                     df_test.loc[mask, 'Fatalities'] = next_fatal_scaled

#         # Fill cases

#         lowest_pred = np.max(df_train[df_train['Province_State'] == state]['ConfirmedCases'].values)

#         cases = handle_predictions (df_test[df_test['Province_State'] == state]['ConfirmedCases'].values, lowest_pred)

#         submission = fillSubmission (state, 'ConfirmedCases', cases)

#         # Fill fatal

#         lowest_pred = np.max(df_train[df_train['Province_State'] == state]['Fatalities'].values)

#         cases = handle_predictions (df_test[df_test['Province_State'] == state]['Fatalities'].values, lowest_pred)

#         submission = fillSubmission (state, 'Fatalities', cases)

#         # Update progress bar

#         pbar.update(1)

        

# print('Time spent for predicting everything was {} minutes'.format(round((time.time()-start_time)/60,1)))

# avg_rmsle()
# cases = []

# fatal = []

# for a in random_validation_set:

#     score = rmsle(a)

#     cases.append(score[0])

#     fatal.append(score[1])

#     print(score)

# print (avg_rmsle())

# print ("Average = {}, {}".format(np.average(cases), np.average(fatal)))
# for a in random_validation_set:

#     plotStatus([a])
# submission = df_test[['ForecastId'] + output_cols]

# submission
# submission.to_csv("submission.csv", index=False)