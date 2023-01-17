# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import packages
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
data = pd.read_csv('/kaggle/input/stockdata/GLOBAL DATA/BRENT.csv', index_col=['Dates'], parse_dates=['Dates'])
data.tail()
brentdata=data.copy()
brentdata
plt.figure(figsize=(18, 6))
plt.plot(brentdata.Close)
plt.title('Closing Price')
plt.grid(True)
plt.show()
brentdata['open_value']=brentdata.Open
brentdata['close_value']=brentdata.Close
brentdata['high_value']=brentdata.High
brentdata['low_value']=brentdata.Low
brentdata['volume_value']=brentdata.Volume

del brentdata['Open']
del brentdata['High']
del brentdata['Low']
del brentdata['Close']
del brentdata['Volume']
brentdata['ret_open_value'] = brentdata.open_value.pct_change(1).mul(100)
brentdata['ret_close_value'] = brentdata.close_value.pct_change(1).mul(100)
brentdata['ret_high_value'] = brentdata.high_value.pct_change(1).mul(100)
brentdata['ret_Low_value'] = brentdata.low_value.pct_change(1).mul(100)
brentdata['ret_volume_value'] = brentdata.volume_value.pct_change(1).mul(100)
brentdata
del brentdata['close_value']
del brentdata['open_value']
del brentdata['low_value']
del brentdata['high_value']
del brentdata['volume_value']
brentdata
brentdata.isna().sum()
brentdata=brentdata.fillna(method='bfill')
brentdata
brentdata.index = pd.DatetimeIndex(brentdata.index).date
start_date = dt.date(2020,1,1)
brentdata = brentdata[start_date : ]
brentdata
trainbrent = brentdata[ : int(0.8*(len(brentdata)))].copy()
model = VAR(endog = trainbrent)
print(model.select_order(trend = 'c'))
model_fit = model.fit(ic = 'aic', trend = 'c')
num_lag = model_fit.k_ar
num_lag
model_fit.summary()
model_fit.forecast(y = trainbrent.values, steps = 1)
def predict(data, fitted_model, lag_order, predict_steps):
    # empty list for our predictions
    prediction = []
  
    # for loop to iterate fitted_model over data
    for i in range(lag_order, len(brentdata)):
        # window of lagged data that the model uses to predict next observation
        window = brentdata.iloc[i - lag_order : i].copy()
        # results of fitted_model being applied to window
        results = fitted_model.forecast(y = window.values, steps = predict_steps)
        # append results to prediction list
        prediction.append(results)
        
    # convert prediction (which is a list of numpy arrays) to a dataframe
    df = np.vstack(prediction)
    df = pd.DataFrame(df)
    # df column names from data
    df.columns = list(brentdata.columns)
    # df index from data
    df.index = brentdata.iloc[len(brentdata) - len(prediction) :].index
    
    # return df
    return df
# root mean squared error
def rmse(predicted, actual):
    # formula for rmse
    residual = predicted - actual
    residual_sq = residual ** 2
    mean_sq = np.mean(residual_sq)
    rmse_value = np.sqrt(mean_sq)
    # return rmse_value
    return rmse_value

# mean absolute error
def mae(predicted, actual):
    # formula for mae
    absolute_residual = np.absolute(predicted - actual)
    mae_value = np.mean(absolute_residual)
    # return mae_value
    return mae_value
def model_graphs(predicted, actual, title = str):
    # RMSE
    rmse_value = rmse(predicted = predicted, actual = actual)
    # MAE
    mae_value = mae(predicted = predicted, actual = actual)
    # start_year (for putting in text box)
    start_year = predicted.iloc[ : 1].index.copy()
    # text box in line plot
    text_str = 'RMSE = ' + str(rmse_value) + '\n MAE = ' + str(mae_value)
    # line plot
    plt.figure(1)
    plt.plot(actual, color = 'blue', linewidth = 2, label = 'actual')
    plt.plot(predicted, color = 'red', linewidth = 1, label = 'predicted')
    plt.legend()
    plt.title(title + ' Actual vs Predicted')
    plt.text(x = start_year, y = 0.2, s = text_str)
    # residual & hist
    plt.figure(2)
    residual = actual - predicted
    plt.hist(residual, bins = 200)
    plt.title('Distribution of ' + title + ' residual')
    plt.axvline(residual.mean(), color = 'k', linestyle = 'dashed', linewidth = 1)
    # show graphics
    plt.show()
def category(x):
    if x >= 0:
        return 'up'
    elif x < 0:
        return 'down'

# function that returns confusion matrix of model with metrics
def confusion_matrix(predicted, actual, title = str):
    df = pd.DataFrame()
    df['predicted'] = predicted.apply(category)
    df['actual'] = actual.apply(category)
    # code
    df.loc[(df['predicted'] == 'up') & (df['actual'] == 'up'), 'code'] = 'true_positive'
    df.loc[(df['predicted'] == 'up') & (df['actual'] == 'down'), 'code'] = 'false_positive'
    df.loc[(df['predicted'] == 'down') & (df['actual'] == 'down'), 'code'] = 'true_negative'
    df.loc[(df['predicted'] == 'down') & (df['actual'] == 'up'), 'code'] = 'false_negative'
    # confusion dictionary
    z = dict(df['code'].value_counts())
    # confusion metrics
    accuracy = (z['true_positive'] + z['true_negative']) / (z['true_positive'] + z['true_negative'] + z['false_positive'] + z['false_negative'])
    true_positive_rate = z['true_positive'] / (z['true_positive'] + z['false_negative'])
    false_positive_rate = z['false_positive'] / (z['false_positive'] + z['true_negative'])
    true_negative_rate = z['true_negative'] / (z['true_negative'] + z['false_positive'])
    false_negative_rate = z['false_negative'] / (z['false_negative'] + z['true_positive'])
    # print metrics
    print('\nMetrics for [{0}]\nAccuracy:{1:6.3f} \nTP Rate:{2:7.3f} \nFP Rate:{3:7.3f}\nTN Rate:{4:7.3f} \nFN Rate:{5:7.3f}'.format(str(title), accuracy, true_positive_rate, false_positive_rate, true_negative_rate, false_negative_rate))
    # print confusion matrix graph
    print('\n'+
      '            [{title}] Confusion Matrix\n'.format(title = str(title))+
      '\n'+
      '           |-------------|-------------|\n'+
      '  n= {0}  | Predicted:  | Predicted:  |\n'.format(z['true_positive']+z['false_positive']+z['true_negative']+z['false_negative'])+
      '           |    Down     |    Up       |\n'+
      '|----------|-------------|-------------|------------|\n'+
      '| Actual:  |             |             |            |\n'+
      '|  Down    |  tn: {0}    |  fp: {1}    |    {2}     |\n'.format(z['true_negative'], z['false_positive'], z['true_negative']+z['false_positive'])+
      '|----------|-------------|-------------|------------|\n'+
      '| Actual:  |             |             |            |\n'+
      '|   UP     |  fn: {0}    |  tp: {1}    |    {2}    |\n'.format(z['false_negative'], z['true_positive'] ,z['false_negative']+z['true_positive'])+
      '|----------|-------------|-------------|------------|\n'+
      '           |             |             |\n'+
      '           |      {0}    |      {1}   |\n'.format(z['true_negative']+z['false_negative'], z['false_positive']+z['true_positive'])+
      '           |-------------|-------------|\n')
    # return df
    return df
train_predicted = model_fit.fittedvalues.copy()
train_actual = trainbrent.iloc[num_lag : len(trainbrent)]
model_graphs(predicted = train_predicted['ret_close_value'], actual = train_actual['ret_close_value'], title = 'Training')
train_confusion = confusion_matrix(predicted = train_predicted['ret_close_value'], actual = train_actual['ret_close_value'], title = 'Train')
test_lag = brentdata.iloc[len(trainbrent) - num_lag :]
test_predicted = predict(data = test_lag, fitted_model = model_fit, lag_order = num_lag, predict_steps = 1)
test_actual = brentdata.iloc[len(trainbrent) :]
model_graphs(predicted = test_predicted['ret_close_value'], actual = test_actual['ret_close_value'], title = 'Test')
persistent_predicted = brentdata.shift(1)
persistent_predicted = persistent_predicted.iloc[len(trainbrent) : ]
persistent_actual = test_actual.copy()
model_graphs(predicted = persistent_predicted['ret_close_value'], actual = persistent_actual['ret_close_value'], title = 'Persistent')
persistent_confusion = confusion_matrix(predicted = persistent_predicted['ret_close_value'], actual = persistent_actual['ret_close_value'], title = 'Persistent')
