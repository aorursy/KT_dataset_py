# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import multiprocessing

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448

mem_gib = mem_bytes/(1024.**3)  # e.g. 3.74

print("RAM: %f GB" % mem_gib)

print("CORES: %d" % multiprocessing.cpu_count())



# Any results you write to the current directory are saved as output.
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from tqdm import tqdm

import time

from datetime import datetime

from pathlib import Path

from sklearn import preprocessing

import keras.backend as K

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout

from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler, MinMaxScaler
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

train.tail()
test.tail()
train.info()
mask = train['Date'].max()

world_cum_confirmed = sum(train[train['Date'] == mask].ConfirmedCases)

world_cum_fatal = sum(train[train['Date'] == mask].Fatalities)
print('Number of Countires are: ', len(train['Country_Region'].unique()))

print('Training dataset ends at: ', mask)

print('Number of cumulative confirmed cases worldwide are: ', world_cum_confirmed)

print('Number of cumulative fatal cases worldwide are: ', world_cum_fatal)
# top 10 countires that have most servere situation

cum_per_country = train[train['Date'] == mask].groupby(['Date','Country_Region']).sum().sort_values(['ConfirmedCases'], ascending=False)

cum_per_country[:10]
# plot growing curve for top 5 most servere countries except China

#TODO: optimize code

date = train['Date'].unique()

cc_us = train[train['Country_Region'] == 'US'].groupby(['Date']).sum().ConfirmedCases

ft_us = train[train['Country_Region'] == 'US'].groupby(['Date']).sum().Fatalities

cc_ity = train[train['Country_Region'] == 'Italy'].groupby(['Date']).sum().ConfirmedCases

ft_ity = train[train['Country_Region'] == 'Italy'].groupby(['Date']).sum().Fatalities

cc_spn = train[train['Country_Region'] == 'Spain'].groupby(['Date']).sum().ConfirmedCases

ft_spn = train[train['Country_Region'] == 'Spain'].groupby(['Date']).sum().Fatalities

cc_gmn = train[train['Country_Region'] == 'Germany'].groupby(['Date']).sum().ConfirmedCases

ft_gmn = train[train['Country_Region'] == 'Germany'].groupby(['Date']).sum().Fatalities

cc_frc = train[train['Country_Region'] == 'France'].groupby(['Date']).sum().ConfirmedCases

ft_frc = train[train['Country_Region'] == 'France'].groupby(['Date']).sum().Fatalities



fig = go.Figure()

# add traces

fig.add_trace(go.Scatter(x=date, y=cc_us, name='US'))

fig.add_trace(go.Scatter(x=date, y=cc_ity, name='Italy'))

fig.add_trace(go.Scatter(x=date, y=cc_spn, name='Spain'))

fig.add_trace(go.Scatter(x=date, y=cc_gmn, name='Germany'))

fig.add_trace(go.Scatter(x=date, y=cc_frc, name='France'))

fig.update_layout(title="Plot of Cumulative Cases for Top 5 countires (except China)",

    xaxis_title="Date",

    yaxis_title="Cases")

fig.update_xaxes(nticks=30)



fig.show()
fig = go.Figure()

# add traces

fig.add_trace(go.Scatter(x=date, y=ft_us, name='US'))

fig.add_trace(go.Scatter(x=date, y=ft_ity, name='Italy'))

fig.add_trace(go.Scatter(x=date, y=ft_spn, name='Spain'))

fig.add_trace(go.Scatter(x=date, y=ft_gmn, name='Germany'))

fig.add_trace(go.Scatter(x=date, y=ft_frc, name='France'))

fig.update_layout(title="Plot of Fatal Cases for Top 5 countires (except China)",

    xaxis_title="Date",

    yaxis_title="Cases")

fig.update_xaxes(nticks=30)



fig.show()
#TODO: check duplicates,missing numeric, string, typo.
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])

train['Country_Region'] = train['Country_Region'].astype(str)

# train['Province_State'] = train['Province_State'].astype(str)

test['Country_Region'] = test['Country_Region'].astype(str)

# test['Province_State'] = test['Province_State'].astype(str)
EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state





train['Province_State'].fillna(EMPTY_VAL, inplace=True)

train['Province_State'] = train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)



test['Province_State'].fillna(EMPTY_VAL, inplace=True)

test['Province_State'] = test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
le = preprocessing.LabelEncoder()

train['country_encoder'] = le.fit_transform(train['Country_Region'])

train['date_int'] = train['Date'].apply(lambda x: datetime.strftime(x, '%m%d')).astype(int)



test['country_encoder'] = le.transform(test['Country_Region'])

test['date_int'] = test['Date'].apply(lambda x: datetime.strftime(x, '%m%d')).astype(int)
le = preprocessing.LabelEncoder()

train['province_encoder'] = le.fit_transform(train['Province_State'])

test['province_encoder'] = le.transform(test['Province_State'])
# #TODO: takes 44m ish, consider multi-processing, multi-cores, run in GPU

# #TODO: create data_generate func

start_time = time.time()



country = train['Country_Region'].drop_duplicates()

train_df = train.copy()

train_df.rename(columns={'Date': 'date', 'ConfirmedCases': 'cc_cases', 'Fatalities': 'ft_cases', 'Country_Region': 'country', 'Province_State': 'province'}, inplace=True)

lags = np.arange(1,8,1)  # lag of 1 to 7



with tqdm(total = len(list(train_df['date'].unique()))) as pbar:

    for d in train_df['date'].drop_duplicates():

        for i in country:

            province = train_df[train_df['country'] == i]['province'].drop_duplicates()

            for j in province:

                mask = (train_df['date'] == d) & (train_df['country'] == i) & (train_df['province'] == j)            

                for lag in lags:

                    mask_org = (train_df['date'] == (d - pd.Timedelta(days=lag))) & (train_df['country'] == i) & (train_df['province'] == j)

                    try:

                        train_df.loc[mask, 'cc_cases_' + str(lag)] = train_df.loc[mask_org, 'cc_cases'].values

                    except:

                        train_df.loc[mask, 'cc_cases_' + str(lag)] = 0



                    try:

                        train_df.loc[mask, 'ft_cases_' + str(lag)] = train_df.loc[mask_org, 'ft_cases'].values

                    except:

                        train_df.loc[mask, 'ft_cases_' + str(lag)] = 0

        pbar.update(1)

print('Time spent for building features is {} minutes'.format(round((time.time()-start_time)/60,1)))
train_df.to_csv(Path('/kaggle/working', 'train_df.csv')) 

# saved locally, reload it

#train_df = pd.read_csv(Path('/kaggle/input/covid19-global-forecasting-week-2', 'train.csv'), index_col = 0, parse_dates = ['date'])

train_df[train_df['country'] == 'Italy'].tail()
#TODO: walk forward validation

def split_train_val(df, val_ratio):

    val_len = int(len(df) * val_ratio)

    train_set =  df[:-val_len]

    val_set = df[-val_len:]

    return train_set, val_set
test_fixed_cols = ['ForecastId', 'Province_State', 'Country_Region', 'Date']

fixed_cols = ['Id', 'province', 'country', 'date']

output_cols = ['cc_cases', 'ft_cases']

input_cols = list(set(train_df.columns.to_list()) - set(fixed_cols) - set(output_cols))

print('output columns are ', output_cols)

print('input columns are ', input_cols)

X = train_df[input_cols]

y = train_df[output_cols]
# split to cumulative and fatal features and build 2 separate models

# split to train and validation set

cc_input = ['cc_cases_1', 'cc_cases_2', 'cc_cases_3', 'cc_cases_4', 'cc_cases_5', 'cc_cases_6', 'cc_cases_7', 'country_encoder', 'province_encoder', 'date_int']

ft_input = ['ft_cases_1', 'ft_cases_2', 'ft_cases_3', 'ft_cases_4', 'ft_cases_5', 'ft_cases_6', 'ft_cases_7', 'country_encoder', 'province_encoder', 'date_int']

cc_output = ['cc_cases']

ft_output = ['ft_cases']

X_cc = X[cc_input]

X_ft = X[ft_input]

y_cc = y[cc_output]

y_ft = y[ft_output]

train_X_cc, val_X_cc = split_train_val(df = X_cc, val_ratio = 0.1)

train_y_cc, val_y_cc = split_train_val(df = y_cc, val_ratio = 0.1)

train_X_ft, val_X_ft = split_train_val(df = X_ft, val_ratio = 0.1)

train_y_ft, val_y_ft = split_train_val(df = y_ft, val_ratio = 0.1)
idx = np.random.RandomState(seed=42).permutation(train_X_cc.index)

train_X_cc = train_X_cc.reindex(idx)

train_y_cc = train_y_cc.reindex(idx)

train_X_ft = train_X_ft.reindex(idx)

train_y_ft = train_y_ft.reindex(idx)

# train_y_cc.tail()
# # normalization

X_scaler_cc = MinMaxScaler()

X_train_cc = X_scaler_cc.fit_transform(train_X_cc)

X_val_cc =  X_scaler_cc.transform(val_X_cc) # intput/output 2D array-like



y_scaler_cc = MinMaxScaler()

y_train_cc = y_scaler_cc.fit_transform(train_y_cc)

y_val_cc = y_scaler_cc.transform(val_y_cc) # array-like
X_scaler_ft = MinMaxScaler()

X_train_ft = X_scaler_ft.fit_transform(train_X_ft)

X_val_ft =  X_scaler_ft.transform(val_X_ft) # intput/output 2D array-like



y_scaler_ft = MinMaxScaler()

y_train_ft = y_scaler_ft.fit_transform(train_y_ft)

y_val_ft = y_scaler_ft.transform(val_y_ft) # array-like
print('Validate if train and test is splited correctly for 2 cases: ')

print('cumulative cases training has shape ', X_train_cc.shape, y_train_cc.shape)

print('fatal cases training has shape ', X_train_ft.shape, y_train_ft.shape)

print('cumulative cases valid has shape ', X_val_cc.shape, y_val_cc.shape)

print('fatal cases valid has shape ', X_val_ft.shape, y_val_ft.shape)

# #TODO

print('Validate if train and test contains np.nan, np.inf, -np.inf after standardization: ')
'''# if choose to not apply normalization, however it generates NaN in output...

X_train_cc = train_X_cc.to_numpy()  

X_val_cc = val_X_cc.to_numpy()

X_train_ft = train_X_ft.to_numpy()

X_val_ft = val_X_ft.to_numpy()



y_train_cc = train_y_cc.to_numpy()

y_val_cc = val_y_cc.to_numpy()

y_train_ft = train_y_ft.to_numpy()

y_val_ft = val_y_ft.to_numpy()'''
# for LSTM, intput.shape = (n_samples, 1, n_features)

X_train_cc = X_train_cc.reshape(X_train_cc.shape[0], 1, X_train_cc.shape[1])

X_val_cc = X_val_cc.reshape(X_val_cc.shape[0], 1, X_val_cc.shape[1])



X_train_ft = X_train_ft.reshape(X_train_ft.shape[0], 1, X_train_ft.shape[1])

X_val_ft = X_val_ft.reshape(X_val_ft.shape[0], 1, X_val_ft.shape[1])

print(X_train_cc.shape, X_val_cc.shape, X_train_ft.shape, X_val_ft.shape)
# customize loss function which is aligned with kaggle evaluation

def root_mean_squared_log_error(y_true, y_pred):

    return K.sqrt(K.mean(K.square(K.log(y_pred + 1) - K.log(y_true + 1)))) 
#declaring only one model

def LSTM_model(n_1, input_dim, output_dim):

    model = Sequential()

    model.add(LSTM(n_1,input_shape=(1, input_dim), activation='relu'))

    model.add(Dropout(0.1))

    # model.add(LSTM(n_2, activation='relu'))

    model.add(Dense(output_dim, activation='relu'))

    model.compile(loss=root_mean_squared_log_error, optimizer='adam')

    print(model.summary())

    return model
#TODO: debug sometimes it's getting inf. Suspect bad input

model_cc = LSTM_model(4, X_train_cc.shape[-1], y_train_cc.shape[-1])

model_ft = LSTM_model(4, X_train_ft.shape[-1], y_train_ft.shape[-1])

early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0, mode='min')
print('Start model training')

start_time = time.time()

history_cc = model_cc.fit(X_train_cc, y_train_cc, epochs = 100,validation_data = (X_val_cc, y_val_cc), verbose = 2, callbacks=[early_stop])

model_cc.save("model_cc.h5")

print('Time spent for model training is {} minutes'.format(round((time.time()-start_time)/60,1)))
# Plot training & validation loss values

plt.figure(figsize=(8,5))

plt.plot(history_cc.history['loss'])

plt.plot(history_cc.history['val_loss'])

plt.title('CC Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
print('Start model training')

start_time = time.time()

history_ft = model_ft.fit(X_train_ft, y_train_ft, epochs = 100,validation_data = (X_val_ft, y_val_ft), verbose = 2, callbacks=[early_stop])

model_ft.save("model_ft.h5")

print('Time spent for model training is {} minutes'.format(round((time.time()-start_time)/60,1)))
# Plot training & validation loss values

plt.figure(figsize=(8,5))

plt.plot(history_ft.history['loss'])

plt.plot(history_ft.history['val_loss'])

plt.title('FT Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Validate if output makes sense

yhat_val_cc = model_cc.predict(X_val_cc)

print(yhat_val_cc)
print(val_y_cc)
# Validate if output makes sense

yhat_val_ft = model_cc.predict(X_val_ft)

print(yhat_val_ft)
print(val_y_ft)
#TODO: takes 14m ish, consider multi-processing, multi-cores, run in GPU

#TODO: create data_generate func

start_time = time.time()

test['Country_Region'] = test['Country_Region'].astype(str)

test['Province_State'] = test['Province_State'].astype(str)

country = test['Country_Region'].drop_duplicates()

adj_input_cols = [e for e in input_cols if e not in ('province_encoder', 'country_encoder', 'date_int')]

# fill data for overlapped days

test_df = test.copy().join(pd.DataFrame(columns = adj_input_cols + output_cols))

test_df.rename(columns={'Date': 'date', 'Country_Region': 'country', 'Province_State': 'province'}, inplace=True)

lags = np.arange(1,8,1)  # lag of 1 to 7

test_overlap_mask = (test_df['date'] <= train_df['date'].max())

train_overlap_mask = (train_df['date'] >= test_df['date'].min())

test_df.loc[test_overlap_mask, input_cols + output_cols] = train_df.loc[train_overlap_mask, input_cols + output_cols].values



# predict data for forward days

pred_dt_range = pd.date_range(start = train_df['date'].max() + pd.Timedelta(days=1), end = test_df['date'].max(), freq = '1D') # test_df['date'].max()

with tqdm(total = len(pred_dt_range)) as pbar:

    for d in pred_dt_range:

        

        for i in country:

            

            province = test_df[test_df['country'] == i]['province'].drop_duplicates()

            

            for j in province:

                

                mask = (test_df['date'] == d) & (test_df['country'] == i) & (test_df['province'] == j)

                

                

                # update input features for the predicted day

                for lag in lags:

                    mask_org = (test_df['date'] == (d - pd.Timedelta(days=lag))) & (test_df['country'] == i) & (test_df['province'] == j)

                    try:

                        test_df.loc[mask, 'cc_cases_' + str(lag)] = test_df.loc[mask_org, 'cc_cases'].values

                    except:

                        test_df.loc[mask, 'cc_cases_' + str(lag)] = 0



                    try:

                        test_df.loc[mask, 'ft_cases_' + str(lag)] = test_df.loc[mask_org, 'ft_cases'].values

                    except:

                        test_df.loc[mask, 'ft_cases_' + str(lag)] = 0

                

                test_X  = test_df.loc[mask, input_cols]

            

                # predict for comfirmed cases

                test_X_cc = test_X[cc_input]

                X_test_cc= test_X_cc

                # X_test_cc =  X_scaler_cc.transform(test_X_cc) # intput/output 2D array-like

                # X_test_cc = X_test_cc.reshape(X_test_cc.shape[0], 1, X_test_cc.shape[1])

                X_test_cc = X_test_cc.to_numpy().reshape(X_test_cc.shape[0], 1, X_test_cc.shape[1])

                next_cc = model_cc.predict(X_test_cc)

                # next_cc_scaled = y_scaler_cc.inverse_transform(next_cc)

                next_cc_scaled = next_cc

                

                # predict for fatal cases

                test_X_ft = test_X[ft_input]

                X_test_ft = test_X_ft

                # X_test_ft =  X_scaler_ft.transform(test_X_ft) # intput/output 2D array-like

                # X_test_ft = X_test_ft.reshape(X_test_ft.shape[0], 1, X_test_ft.shape[1])

                X_test_ft = X_test_ft.to_numpy().reshape(X_test_ft.shape[0], 1, X_test_ft.shape[1])

                next_ft = model_cc.predict(X_test_ft)

                # next_ft_scaled = y_scaler_cc.inverse_transform(next_ft)

                next_ft_scaled = next_ft

                # print(d, ' - ', i, ' - ', j,  ' - Predicted Confirmed Cases are ', next_cc_scaled, ' - Predicted Fatal Cases are ', next_ft_scaled)

                

                # update yhat for next day

                test_df.loc[mask, 'cc_cases'] = next_cc_scaled

                test_df.loc[mask, 'ft_cases'] = next_ft_scaled

                        

        pbar.update(1)

        

print('Time spent for building features is {} minutes'.format(round((time.time()-start_time)/60,1)))
submission = pd.DataFrame()

submission['ForecastId'] = test_df['ForecastId']

submission['ConfirmedCases'] = test_df['cc_cases']

submission['Fatalities'] = test_df['ft_cases']
submission.to_csv("submission.csv",index=False)
submission[:20]