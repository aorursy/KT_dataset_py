import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time

from datetime import datetime
from collections import Counter

from pandas.plotting import autocorrelation_plot as acp
from statsmodels.graphics.tsaplots import plot_pacf
uber_jan_june = pd.read_csv('../input/uber-raw-data-janjune-15.csv')
print (uber_jan_june.head())
uber_jan_june.shape
## Extracting month and day from date-time
uber_jan_june['Month_Day'] = uber_jan_june['Pickup_date'].apply(lambda pickup: datetime.strptime(pickup, '%Y-%m-%d %H:%M:%S').strftime('%m-%d').split('-'))
## Separate month and day
uber_jan_june['Month'] = [month_day [0] for month_day in uber_jan_june['Month_Day']]
uber_jan_june['Day'] = [month_day [1] for month_day in uber_jan_june['Month_Day']]
uber_jan_june.tail(20)
jan_june_grouped = uber_jan_june.groupby(by = ['Month', 'Day']).size().unstack()
jan_june_grouped
## Aggregate results to form a time-series
all_jan_june = [jan_june_grouped.iloc[r,:] for r in range(jan_june_grouped.shape[0])]
all_jan_june = [trips for month in all_jan_june for trips in month]
len(all_jan_june)
## Remove missing values: here missing values are the days when a month is shorter than 31 days.
remove_inds = list(np.argwhere(np.isnan(all_jan_june) == True).reshape((1,5))[0])
all_jan_june_mod = [all_jan_june[i] for i,j in enumerate(all_jan_june) if i not in remove_inds]
## Convert time-series into data-frame for modeling process
uber_jan_june_final = pd.DataFrame({'Days': range(1,len(all_jan_june_mod)+1), 'Trips': all_jan_june_mod})
uber_jan_june_final.head()
from sklearn.preprocessing import MinMaxScaler

## Split into train-test set:
train_jan_june = uber_jan_june_final.iloc[0:167,1:2].values
test_jan_june = uber_jan_june_final.iloc[167:,1:2].values

print ('Training data: ', train_jan_june.shape)
print ('Testing data: ', test_jan_june.shape)

## Feature-scaling:
mms = MinMaxScaler(feature_range = (0,1))
train_jan_june_scaled = mms.fit_transform(train_jan_june)
x_train = []
y_train = []

for rides in range(14, 167):
    x_train.append(train_jan_june_scaled[rides-14:rides,0])
    y_train.append(train_jan_june_scaled[rides,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, newshape = (x_train.shape[0], x_train.shape[1], 1))
## Import required modules:
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import GridSearchCV
np.random.seed(11)
t_start = time.time()

def build_rnn(num_units, input_x, input_y, drpout, epochs, size_of_batch, optimizer, loss):
    
    regressor = Sequential()

    ## Adding first LSTM layer:
    regressor.add(LSTM(units = num_units, return_sequences = True, input_shape = (input_x.shape[1],1)))
    regressor.add(Dropout(drpout))

    ## Adding second LSTM layer:
    regressor.add(LSTM(units = num_units, return_sequences = True))
    regressor.add(Dropout(drpout))

    ## Adding third LSTM layer:
    regressor.add(LSTM(units = num_units, return_sequences = True))
    regressor.add(Dropout(drpout))

    ## Adding fourth LSTM layer:
    regressor.add(LSTM(units = num_units, return_sequences = True))
    regressor.add(Dropout(drpout))

    ## Adding fifth LSTM layer:
    regressor.add(LSTM(units = num_units, return_sequences = False))
    regressor.add(Dropout(drpout))

    ## Adding o/p layer:
    regressor.add(Dense(units = 1))

    ## Compiling RNN:
    regressor.compile(optimizer = optimizer, loss = loss)

    ## Fitting RNN to training set:
    regressor.fit(x = input_x, y = input_y, epochs = epochs, batch_size = size_of_batch)

    return regressor
    
regressor = build_rnn(num_units = 40, input_x = x_train, input_y = y_train, drpout = 0.2, epochs = 1000, size_of_batch = 16, optimizer = 'adam', loss = 'mean_squared_error')

print (time.time() - t_start)
adjusted_inputs = uber_jan_june_final[len(uber_jan_june_final) - len(test_jan_june) - 14:]['Trips'].values
adjusted_inputs = adjusted_inputs.reshape(-1,1)
adjusted_inputs = mms.transform(adjusted_inputs)
adjusted_inputs[0:10]
## Create properly structured test set:
x_test = []
for rides in range(14,29):
    x_test.append(adjusted_inputs[rides-14:rides,0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, newshape = (x_test.shape[0], x_test.shape[1], 1))
x_test.shape
## Make prediction for test set and bring values back to original scale
pred = regressor.predict(x_test)
pred = mms.inverse_transform(pred)

## Check RMSE on test-set
residuals = pred[0:-1] - test_jan_june
rmse = np.sqrt(np.mean(residuals**2))
rmse
fig, ax = plt.subplots(figsize = (12,6))

e = [i*0.05 for i in pred]
ax.plot(pred, color = 'red', label = 'Predictions')
ax.errorbar(x = range(15), y = pred, yerr = e, fmt = '*', color = 'r')
ax.plot(test_jan_june, color = 'black', label = 'True')

ax.set_xlabel('Time-steps (Last two weeks of June-2015)', fontsize = 15)
ax.set_ylabel('#Uber-trips', fontsize = 15)
ax.set_title('Comparing LSTM-predictions with Test-data \n RMSE: {}'.format(np.round(rmse,2)), fontsize = 20)

ax.legend()
plt.show()
fig ,ax = plt.subplots(figsize = (12,6))
ax.plot(residuals)

ax.set_xlabel('Days in test-set (last two weeks of June-2015)', fontsize = 15)
ax.set_ylabel('Error in predictions', fontsize = 15)
ax.set_title('Testing Data RMSE w/ LSTM: {}'.format(round(rmse, 3)), fontsize = 20)
plt.show()
pred_train = regressor.predict(x_train)
pred_train = mms.inverse_transform(pred_train)

residuals_train = pred_train - train_jan_june[0:-14]
rmse_train = np.sqrt(np.mean(residuals_train**2))

fig, ax = plt.subplots(figsize = (14,6))
ax.plot(residuals_train)
ax.set_xlabel('Sequence of days (Jan - June 2015)', fontsize = 15)
ax.set_ylabel('Error in predictions', fontsize = 15)
ax.set_title('Training Data RMSE w/ LSTM: {}'.format(round(rmse_train,3)), fontsize = 20)
plt.show()
fig, ax = plt.subplots(figsize = (16,6))

ax.plot(pred_train, color = 'red', label = 'Predictions')
ax.plot(train_jan_june, color = 'black', label = 'True')

ax.axvline(x = 26, color = 'y', linestyle = 'dashed')
ax.text(x = 29, y = 30000, s = 'Monday-01/26 \n Nothing Special')

ax.axvline(x = 135, color = 'y', linestyle = 'dashed')
ax.text(x = 115, y = 30000, s = 'Friday-05/15 \n Nothing Special')

ax.set_xlabel('Time-steps (Jan-June 2015)', fontsize = 15)
ax.set_ylabel('#Uber trips', fontsize = 15)
ax.set_title('Comparing LSTM-predictions with training data \n RMSE: {}'.format(round(rmse_train,3)), fontsize = 20)

ax.legend()
plt.show()

import matplotlib.gridspec as gridspec
plt.figure(figsize = (20,12))
g = gridspec.GridSpec(2,2)

ax1 = plt.subplot(g[0,0])
ax1.hist(residuals_train, normed = True)
pd.DataFrame(residuals_train).plot(kind = 'kde', ax = ax1, label = 'KDE')
ax1.set_title('KDE and Histogram of Residuals-Training', fontsize = 18)
ax1.legend_.remove()

ax2 = plt.subplot(g[0,1])
import scipy.stats as ss
ss.probplot(residuals_train[:,0].tolist(), plot = ax2)
ax2.set_title('Q-Q Plot-Training', fontsize = 18)

ax3 = plt.subplot(g[1,0])
ax3.plot(residuals_train)
ax3.set_xlabel('Sequence of days (Jan - June 2015)', fontsize = 15)
ax3.set_ylabel('Error in predictions', fontsize = 15)
ax3.set_title('Residuals ', fontsize = 20)

ax4 = plt.subplot(g[1,1])
acp(residuals_train, ax = ax4)

plt.show()

# from pandas.plotting import autocorrelation_plot as acp
# from statsmodels.graphics.tsaplots import plot_pacf
# from statsmodels.tsa.stattools import adfuller

## Do differencing
# all_trips_df_diff = uber_jan_june_final[['Trips']] - uber_jan_june_final[['Trips']].shift()

# from statsmodels.tsa.stattools import adfuller
# train_jan_june = all_trips_df_diff['Trips'][0:167]
# test_jan_june = all_trips_df_diff['Trips'][167:]

# from statsmodels.tsa.arima_model import ARIMA
# train_temp = pd.DataFrame({'original_series': uber_jan_june_final['Trips'][0:167], 'shifted_series': all_trips_df_diff['Trips'][0:167]})

# def fit_ARIMA(train, temp):
#     p_all = []
#     q_all = []
#     rmse_train = []
#     predictions = []
    
#     for p in range(6):        
#         for q in range(6):
#             try: 
#                 model = ARIMA(np.array(train), order = (p,0,q))
#                 results_arima = model.fit()                    
#                 fitted_values = results_arima.fittedvalues

#                 p_all.append(p)
#                 q_all.append(q)

#                 Back_to_original = [temp['shifted_series'][i+1] + temp['original_series'][i] for i in range(temp.shape[0]-1)]

#                 ## Add first element as 0:
#                 Back_to_original.insert(0,0)

#                 back_to_actual_fitted_values = [fitted_values[i+1] + temp['original_series'][i] for i in range(temp.shape[0]-1)]
#                 back_to_actual_fitted_values.insert(0,0)
#                 predictions.append(back_to_actual_fitted_values)
                
#                 rmse = np.sqrt(np.mean((temp['original_series'][1:] - back_to_actual_fitted_values[1:])**2))
#                 rmse_train.append(np.ceil(rmse))

#             except:
#                 pass
                
    
#     grid_search_df = pd.DataFrame({'p': p_all, 'q': q_all, 'RMSE_train': rmse_train, 'Predictions': predictions})
    
#     return grid_search_df

# train_grid_search_df = fit_ARIMA(train_jan_june, train_temp)
# train_grid_search_df.sort_values(by = ['RMSE_train']).head()