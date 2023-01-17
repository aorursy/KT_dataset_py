import warnings

warnings.filterwarnings("ignore")

!pip install --upgrade mlfinlab pyfolio python-binance
import time

import math

import timeit

import os.path





import numpy as np

import pandas as pd

import pyfolio as pf

import mlfinlab as ml





from sklearn.utils import resample

from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, classification_report, confusion_matrix, accuracy_score



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score



from dateutil import parser

from tqdm import tqdm_notebook 

from binance.client import Client

from datetime import timedelta, datetime

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()



import matplotlib.pyplot as plt

%matplotlib inline
#Enter your own API-key here

binance_api_key = user_secrets.get_secret("key")   

#Enter your own API-secret here

binance_api_secret = user_secrets.get_secret("sec") 



### CONSTANTS

binsizes = {"1m": 1, "5m": 5, "1h": 60, "5h":300,"12h":720, "1d": 1440}

batch_size = 750

binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)





### FUNCTIONS

def minutes_of_new_data(symbol, kline_size, data, source):

    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])

    elif source == "binance": old = datetime.strptime('1 Jan 2017', '%d %b %Y')

    if source == "binance": new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')

    return old, new



def get_all_binance(symbol, kline_size, save = False):

    filename = '%s-%s-data.csv' % (symbol, kline_size)

    if os.path.isfile(filename): data_df = pd.read_csv(filename)

    else: data_df = pd.DataFrame()

    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, source = "binance")

    delta_min = (newest_point - oldest_point).total_seconds()/60

    available_data = math.ceil(delta_min/binsizes[kline_size])

    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))

    else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))

    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))

    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    if len(data_df) > 0:

        temp_df = pd.DataFrame(data)

        data_df = data_df.append(temp_df)

    else: data_df = data

    data_df.set_index('timestamp', inplace=True)

    if save: data_df.to_csv(filename)

    print('All caught up..!')

    return data_df
data = get_all_binance('BTCUSDT', '1h', save = True)

data.head()
data = data.reset_index()

new_data = pd.concat([data['timestamp'], data['close'], data['volume']], axis=1)

new_data.columns = ['date', 'price', 'volume']

new_data['price'] = new_data['price'].astype('float64') 

new_data['volume'] = new_data['volume'].astype('float64')

new_data.head()

del data

print('Rows:', new_data.shape[0])
#Get dollar bars

data = ml.data_structures.get_dollar_bars(new_data, threshold=700000, batch_size=10000, verbose=True)

data.index = pd.to_datetime(data['date_time'])

data = data.drop('date_time', axis=1)

data.head()
# compute moving averages

fast_window = 5

slow_window = 10



data['fast_mavg'] = data['close'].rolling(window=fast_window, min_periods=fast_window, center=False).mean()

data['slow_mavg'] = data['close'].rolling(window=slow_window, min_periods=slow_window, center=False).mean()

data.head()



# Compute sides

data['side'] = np.nan



long_signals = data['fast_mavg'] >= data['slow_mavg'] 

short_signals = data['fast_mavg'] < data['slow_mavg'] 

data.loc[long_signals, 'side'] = 1

data.loc[short_signals, 'side'] = -1



# Remove Look ahead biase by lagging the signal

data['side'] = data['side'].shift(1)



# Save the raw data

raw_data = data.copy()



# Drop the NaN values from our data set

data.dropna(axis=0, how='any', inplace=True)

data['side'].value_counts()
# Compute daily volatility

daily_vol = ml.util.get_daily_vol(close=data['close'], lookback=40)



# Apply Symmetric CUSUM Filter and get timestamps for events

# Note: Only the CUSUM filter needs a point estimate for volatility

#Important dates

#start date 2017-08-18

#end date for validation: 2019-12-31

cusum_events = ml.filters.cusum_filter(data['close'], threshold=daily_vol['2017-08-18':'2019-12-31'].mean()*0.5)



# Compute vertical barrier

vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events, close=data['close'], num_days=1)
pt_sl = [1, 2]

min_ret = 0.005

triple_barrier_events = ml.labeling.get_events(close=data['close'],

                                               t_events=cusum_events,

                                               pt_sl=pt_sl,

                                               target=daily_vol,

                                               min_ret=min_ret,

                                               num_threads=3,

                                               vertical_barrier_times=vertical_barriers,

                                               side_prediction=data['side'])
labels = ml.labeling.get_bins(triple_barrier_events, data['close'])

labels.side.value_counts()
primary_forecast = pd.DataFrame(labels['bin'])

primary_forecast['pred'] = 1

primary_forecast.columns = ['actual', 'pred']



# Performance Metrics

actual = primary_forecast['actual']

pred = primary_forecast['pred']

print(classification_report(y_true=actual, y_pred=pred))



print("Confusion Matrix")

print(confusion_matrix(actual, pred))



print('')

print("Accuracy")

print(accuracy_score(actual, pred))
raw_data.head()
# Log Returns

raw_data['log_ret'] = np.log(raw_data['close']).diff()

# Momentum

raw_data['mom1'] = raw_data['close'].pct_change(periods=1)

raw_data['mom2'] = raw_data['close'].pct_change(periods=2)

raw_data['mom3'] = raw_data['close'].pct_change(periods=3)

raw_data['mom4'] = raw_data['close'].pct_change(periods=4)

raw_data['mom5'] = raw_data['close'].pct_change(periods=5)



# Volatility

raw_data['volatility_10'] = raw_data['log_ret'].rolling(window=10, min_periods=10, center=False).std()

raw_data['volatility_6'] = raw_data['log_ret'].rolling(window=6, min_periods=6, center=False).std()

raw_data['volatility_3'] = raw_data['log_ret'].rolling(window=3, min_periods=3, center=False).std()



# Serial Correlation (Takes about 4 minutes)

window_autocorr = 10



raw_data['autocorr_1'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=1), raw=False)

raw_data['autocorr_2'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=2), raw=False)

raw_data['autocorr_3'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=3), raw=False)

raw_data['autocorr_4'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=4), raw=False)

raw_data['autocorr_5'] = raw_data['log_ret'].rolling(window=window_autocorr, min_periods=window_autocorr, center=False).apply(lambda x: x.autocorr(lag=5), raw=False)



# Get the various log -t returns

raw_data['log_t1'] = raw_data['log_ret'].shift(1)

raw_data['log_t2'] = raw_data['log_ret'].shift(2)

raw_data['log_t3'] = raw_data['log_ret'].shift(3)

raw_data['log_t4'] = raw_data['log_ret'].shift(4)

raw_data['log_t5'] = raw_data['log_ret'].shift(5)
# Re-compute sides

raw_data['side'] = np.nan



long_signals = raw_data['fast_mavg'] >= raw_data['slow_mavg']

short_signals = raw_data['fast_mavg'] < raw_data['slow_mavg']



raw_data.loc[long_signals, 'side'] = 1

raw_data.loc[short_signals, 'side'] = -1
# Remove look ahead bias

raw_data = raw_data.shift(3)

raw_data = raw_data.fillna(method='ffill')

raw_data = raw_data.fillna(0)

raw_data.head()
# Get features at event dates

X = raw_data.loc[labels.index, :]



# Drop unwanted columns

X.drop(['open', 'high', 'low', 'close', 'cum_buy_volume', 'cum_dollar_value', 'cum_ticks','fast_mavg', 'slow_mavg',], axis=1, inplace=True)



y = labels['bin']
y.value_counts()
# Split data into training, validation and test sets

X_training_validation = X['2017-08-18':'2019-12-31']

y_training_validation = y['2017-08-18':'2019-12-31']

X_train, X_validate, y_train, y_validate = train_test_split(X_training_validation, y_training_validation, test_size=0.15, shuffle=False)
train_df = pd.concat([y_train, X_train], axis=1, join='inner')

train_df['bin'].value_counts()
# Upsample the training data to have a 50 - 50 split

# https://elitedatascience.com/imbalanced-classes

majority = train_df[train_df['bin'] == 0]

minority = train_df[train_df['bin'] == 1]



new_minority = resample(minority, 

                   replace=True,     # sample with replacement

                   n_samples=majority.shape[0],    # to match majority class

                   random_state=42)



train_df = pd.concat([majority, new_minority])

train_df = shuffle(train_df, random_state=42)



train_df['bin'].value_counts()
# Create training data

y_train = train_df['bin']

X_train= train_df.loc[:, train_df.columns != 'bin']
parameters = {'max_depth':[2, 3, 4, 5, 7],

              'n_estimators':[1, 10, 25, 50, 100, 256, 512],

              'random_state':[42]}

    

def perform_grid_search(X_data, y_data):

    rf = RandomForestClassifier(criterion='entropy')

    clf = GridSearchCV(rf, parameters, cv=4, scoring='roc_auc', n_jobs=3)

    clf.fit(X_data, y_data)

    print(clf.cv_results_['mean_test_score'])

    return clf.best_params_['n_estimators'], clf.best_params_['max_depth']
# extract parameters

n_estimator, depth = perform_grid_search(X_train, y_train)

c_random_state = 42

print(n_estimator, depth, c_random_state)
# Refit a new model with best params, so we can see feature importance

rf = RandomForestClassifier(max_depth=depth, n_estimators=n_estimator,

                            criterion='entropy', random_state=c_random_state)



rf.fit(X_train, y_train.values.ravel())
# Performance Metrics

y_pred_rf = rf.predict_proba(X_train)[:, 1]

y_pred = rf.predict(X_train)

fpr_rf, tpr_rf, _ = roc_curve(y_train, y_pred_rf)

print(classification_report(y_train, y_pred))



print("Confusion Matrix")

print(confusion_matrix(y_train, y_pred))



print('')

print("Accuracy")

print(accuracy_score(y_train, y_pred))



plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_rf, tpr_rf, label='RF')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()
# Meta-label

# Performance Metrics

y_pred_rf = rf.predict_proba(X_validate)[:, 1]

y_pred = rf.predict(X_validate)

fpr_rf, tpr_rf, _ = roc_curve(y_validate, y_pred_rf)

print(classification_report(y_validate, y_pred))



print("Confusion Matrix")

print(confusion_matrix(y_validate, y_pred))



print('')

print("Accuracy")

print(accuracy_score(y_validate, y_pred))



plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_rf, tpr_rf, label='RF')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()
print(X_validate.index.min())

print(X_validate.index.max())
# Feature Importance

title = 'Feature Importance:'

figsize = (15, 5)



feat_imp = pd.DataFrame({'Importance':rf.feature_importances_})    

feat_imp['feature'] = X.columns

feat_imp.sort_values(by='Importance', ascending=False, inplace=True)

feat_imp = feat_imp



feat_imp.sort_values(by='Importance', inplace=True)

feat_imp = feat_imp.set_index('feature', drop=True)

feat_imp.plot.barh(title=title, figsize=figsize)

plt.xlabel('Feature Importance Score')

plt.show()
def get_daily_returns(intraday_returns):

    """

    This changes returns into daily returns that will work using pyfolio. Its not perfect...

    """

    cum_rets = ((intraday_returns + 1).cumprod())

    # Downsample to daily

    daily_rets = cum_rets.resample('B').last()

    # Forward fill, Percent Change, Drop NaN

    daily_rets = daily_rets.ffill().pct_change().dropna()

    return daily_rets
valid_dates = X_validate.index

base_rets = labels.loc[valid_dates, 'ret']

primary_model_rets = get_daily_returns(base_rets)
# Set-up the function to extract the KPIs from pyfolio

perf_func = pf.timeseries.perf_stats
# Save the statistics in a dataframe

perf_stats_all = perf_func(returns=primary_model_rets, 

                           factor_returns=None, 

                           positions=None,

                           transactions=None,

                           turnover_denom="AGB")

perf_stats_df = pd.DataFrame(data=perf_stats_all, columns=['Primary Model'])



pf.show_perf_stats(primary_model_rets)
meta_returns = labels.loc[valid_dates, 'ret'] * y_pred

daily_meta_rets = get_daily_returns(meta_returns)
# Save the KPIs in a dataframe

perf_stats_all = perf_func(returns=daily_meta_rets, 

                           factor_returns=None, 

                           positions=None,

                           transactions=None,

                           turnover_denom="AGB")



perf_stats_df['Meta Model'] = perf_stats_all



pf.show_perf_stats(daily_meta_rets)
# Extarct data for out-of-sample (OOS)

X_oos = X['2019-12-31':]

y_oos = y['2019-12-31':]
# Performance Metrics

y_pred_rf = rf.predict_proba(X_oos)[:, 1]

y_pred = rf.predict(X_oos)

fpr_rf, tpr_rf, _ = roc_curve(y_oos, y_pred_rf)

print(classification_report(y_oos, y_pred))



print("Confusion Matrix")

print(confusion_matrix(y_oos, y_pred))



print('')

print("Accuracy")

print(accuracy_score(y_oos, y_pred))



plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_rf, tpr_rf, label='RF')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()
# Primary model

primary_forecast = pd.DataFrame(labels['bin'])

primary_forecast['pred'] = 1

primary_forecast.columns = ['actual', 'pred']



subset_prim = primary_forecast['2018-01-02':]



# Performance Metrics

actual = subset_prim['actual']

pred = subset_prim['pred']

print(classification_report(y_true=actual, y_pred=pred))



print("Confusion Matrix")

print(confusion_matrix(actual, pred))



print('')

print("Accuracy")

print(accuracy_score(actual, pred))
test_dates = X_oos.index



# Downsample to daily

prim_rets_test = labels.loc[test_dates, 'ret']

daily_rets_prim = get_daily_returns(prim_rets_test)



# Save the statistics in a dataframe

perf_stats_all = perf_func(returns=daily_rets_prim, 

                           factor_returns=None, 

                           positions=None,

                           transactions=None,

                           turnover_denom="AGB")



perf_stats_df['Primary Model OOS'] = perf_stats_all



# pf.create_returns_tear_sheet(labels.loc[test_dates, 'ret'], benchmark_rets=None)

pf.show_perf_stats(daily_rets_prim)
meta_returns = labels.loc[test_dates, 'ret'] * y_pred

daily_rets_meta = get_daily_returns(meta_returns)



# save the KPIs in a dataframe

perf_stats_all = perf_func(returns=daily_rets_meta, 

                           factor_returns=None, 

                           positions=None,

                           transactions=None,

                           turnover_denom="AGB")



perf_stats_df['Meta Model OOS'] = perf_stats_all



pf.create_returns_tear_sheet(daily_rets_meta, benchmark_rets=None)