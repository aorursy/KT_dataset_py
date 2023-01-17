### Loading required libraries

from    pandas             import   read_csv, Grouper, DataFrame, concat
import  matplotlib.pyplot  as       plt
import  statsmodels.api          as       sm
from   sklearn.metrics      import  mean_squared_error
from   statsmodels.tsa.holtwinters     import  SimpleExpSmoothing,Holt, ExponentialSmoothing
import statsmodels.tsa.holtwinters     as      ets
import statsmodels.tools.eval_measures as      fa

import numpy as np
import pandas as pd
from datetime import datetime
from pandas import Series
# Read the time series data
# We set parse_dates to make sure data is read as time-series and make time column as Index
# We drop the id column, which is not usefull for us.
predice = pd.read_csv('/kaggle/input/predice-el-futuro/train_csv.csv', header = 0, index_col = 1, parse_dates = True, squeeze = True)
predice.drop(['id'],axis = 1,inplace = True)
### Print first five records
print(predice.head())
print(predice.tail())
#Clearly, we have no-null values, so no need to worry about missing values
predice.info()
plt.figure(figsize=(12,6))
plt.plot(predice.index, predice['feature'])
plt.title('Feature v/s Time Plot')
plt.xlabel('Time')
plt.ylabel('Feature')
plt.grid(True)
plt.show()
decompPred = sm.tsa.seasonal_decompose(predice.feature, model="additive", freq=4)
decompPred.plot()
plt.show()
def MAE(y,yhat):
    diff = np.abs(np.array(y)-np.array(yhat))
    try:
        mae =  round(np.mean(np.fabs(diff)),3)
    except:
        print("Error while calculating")
        mae = np.nan
    return mae
def MAPE(y, yhat): 
    y, yhat = np.array(y), np.array(yhat)
    try:
        mape =  round(np.sum(np.abs(yhat - y)) / np.sum(y) * 100,2)
    except:
        print("Observed values are empty")
        mape = np.nan
    return mape
train = predice[0:int(len(predice)*0.7)] 
test = predice[int(len(predice)*0.7):]

plt.plot(train.index, train.feature, label = 'Train')
plt.plot(test.index, test.feature,  label = 'Test')
plt.legend(loc = 'best')
plt.title('Original data after split')
plt.show()
rolling = predice['feature'].rolling(window = 2) 
rolling_mean =  rolling.mean()
predice.plot()
rolling_mean.plot(color = 'red')
plt.show()
y_df = pd.DataFrame( {'Observed':predice.feature, 'Predicted':rolling_mean})
y_df .dropna(axis = 0, inplace = True)
print(y_df.tail())

rmse = np.sqrt(mean_squared_error(y_df.Observed, y_df.Predicted))
print("\n\n Accuracy measures ")
print('RMSE: %.3f' % rmse)
n = y_df.shape[0]

mae = MAE(y_df.Observed, y_df.Predicted)
print('MAE: %d' % np.float(mae))

mape = MAPE(y_df.Observed, y_df.Predicted)
print('MAPE: %.3f' % np.float(mape))
# create class
model = SimpleExpSmoothing(np.asarray(train['feature']))

# fit model

alpha_list = [0.1, 0.5, 0.99]

pred_SES  = test.copy() # Have a copy of the test dataset

for alpha_value in alpha_list:

    alpha_str            =  "SES" + str(alpha_value)
    mode_fit_i           =  model.fit(smoothing_level = alpha_value, optimized=False)
    pred_SES[alpha_str]  =  mode_fit_i.forecast(len(test['feature']))
    rmse                 =  np.sqrt(mean_squared_error(test['feature'], pred_SES[alpha_str]))
    mape                 =  MAPE(test['feature'],pred_SES[alpha_str])

    print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse, mape))

# Plotting for alpha = 0.10, as RMSE of it is lowest.
alpha_str = "SES" + str(0.1)
plt.figure(figsize=(12,6))
plt.plot(train.index, train['feature'], label ='Train')
plt.plot(test.index, test['feature'], label  ='Test')
plt.plot(test.index, pred_SES[alpha_str], label  = alpha_str)
plt.title('Simple Exponential Smoothing with alpha : 0.1')
plt.legend(loc='best') 
plt.show()
pred = ExponentialSmoothing(np.asarray(train['feature']),
                                  seasonal_periods= 8, seasonal='additive').fit(optimized=True) 
#After hit-and-trial 

print(pred.params)

print('')
print('== Holt-Winters Additive ETS(A,A,A) Parameters ==')
print('')
alpha_value = np.round(pred.params['smoothing_level'], 4)
print('Smoothing Level: ', alpha_value)
print('Smoothing Slope: ', np.round(pred.params['smoothing_slope'], 4))
print('Smoothing Seasonal: ', np.round(pred.params['smoothing_seasonal'], 4))
print('Initial Level: ', np.round(pred.params['initial_level'], 4))
print('Initial Slope: ', np.round(pred.params['initial_slope'], 4))
print('Initial Seasons: ', np.round(pred.params['initial_seasons'], 4))
print('')
pred_HoltW = test.copy()

pred_HoltW['HoltWM'] = pred.forecast(len(test['feature']))
plt.figure(figsize=(12,6))
plt.plot(train['feature'], label='Train')
plt.plot(test['feature'], label='Test')
plt.plot(pred_HoltW['HoltWM'], label='HoltWinters')
plt.title('Holt-Winters Additive ETS(A,A,M) Parameters:\n  alpha = ' + 
          str(alpha_value) + '  Beta:' + 
          str(np.round(pred.params['smoothing_slope'], 4)) +
          '  Gamma: ' + str(np.round(pred.params['smoothing_seasonal'], 4)))
plt.legend(loc='best')
plt.show()
df_pred_opt =  pd.DataFrame({'Y_hat':pred_HoltW['HoltWM'] ,'Y':test['feature'].values})

rmse_opt    =  np.sqrt(mean_squared_error(df_pred_opt.Y, df_pred_opt.Y_hat))
mape_opt    =  MAPE(df_pred_opt.Y, df_pred_opt.Y_hat)

print("For alpha = %1.2f,  RMSE is %3.4f MAPE is %3.2f" %(alpha_value, rmse_opt, mape_opt))