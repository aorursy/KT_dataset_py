import pandas as pd
import numpy as np
from pandas import Series
from math import sqrt

# metrics
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

# forecasting model
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.arima_model import ARIMA

# for analysis
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from shapely.geometry import LineString

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 7

from IPython.display import display, HTML

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
train_original=pd.read_csv('../input/jetrail-traffic-dataset/Train.csv')
test_original=pd.read_csv('../input/jetrail-traffic-dataset/Test.csv')

train_original.dropna(inplace=True)
test_original.dropna(inplace=True)
test_original.drop(test_original.tail(1).index, inplace=True)

train_df=train_original.copy()
test_df=test_original.copy()
train_original['Datetime']=pd.to_datetime(train_original.Datetime, format='%d-%m-%Y %H:%M')
test_original['Datetime']=pd.to_datetime(test_original.Datetime, format='%d-%m-%Y %H:%M')
train_df['Datetime']=pd.to_datetime(train_df.Datetime, format='%d-%m-%Y %H:%M')
test_df['Datetime']=pd.to_datetime(test_df.Datetime, format='%d-%m-%Y %H:%M')

# generate day, month, year feature
for i in (train_original, test_original, train_df, test_df):
    i['year']=i.Datetime.dt.year
    i['month']=i.Datetime.dt.month
    i['day']=i.Datetime.dt.day
    i['hour']=i.Datetime.dt.hour
# sampling for daily basis
train_df.index=train_df.Datetime
test_df.index=test_df.Datetime

train_df=train_df.resample('D').mean()
test_df=test_df.resample('D').mean()
# split data for training and validation
train=train_df.loc['2012-08-25':'2014-06-24']
valid=train_df.loc['2014-06-25':'2014-09-25']
plt.figure(figsize=(12,7))
train.Count.plot(label='Train')
valid.Count.plot(label='valid')
plt.legend(loc='best')
# determine rolling stats
rolmean=train.Count.rolling(window=7).mean() #for 7 days
rolstd=train.Count.rolling(window=7).std()
rolmean.dropna(inplace=True)
rolstd.dropna(inplace=True)

plt.figure(figsize=(12,7))
rolmean.plot(label='Rolmean', color='green')
rolstd.plot(label='rolstd')
train.Count.plot(label='Train')
plt.legend(loc='best')
# check for stationary
dftest=adfuller(train.Count, autolag='AIC')
dfout=pd.Series(dftest[0:4], index=['Test statistics', 'p-value', '#Lags used', 'Number of observation used'])
for key, val in dftest[4].items():
    dfout['Critical value (%s)'%key]=val

print(dfout)
# Log scale tranformation
# estimating trend
train_count_log=np.log(train.Count)
# train_count_log.plot()
# make TS to be stationary
moving_avg=train_count_log.rolling(window=7).mean()
moving_std=train_count_log.rolling(window=7).std()

train_count_log.plot(label='Log Scale')
moving_avg.plot(label='moving_avg')
moving_std.plot(label='moving_std')
plt.legend(loc='best')
dif_log=train_count_log-moving_avg
dif_log.dropna(inplace=True)
dif_log.plot()
def test_stationary(timeseries):
    # determine roling stats
    mov_avg=timeseries.rolling(window=7).mean()
    mov_std=timeseries.rolling(window=7).std()
    #plot rolling stats
    plt.figure(figsize=(12,7))
    timeseries.plot(label='Original')
    mov_avg.plot(label='Mov avg')
    mov_std.plot(label='Mov std')
    plt.legend(loc='best')
    plt.title('Rolling mean & standard deviation')
    
    # dickey-fuller test
    print('Result of Dickey-fuller test')
    dftest=adfuller(timeseries, autolag='AIC')
    dfout=pd.Series(dftest[:4], index=['Test stats', 'p-value', '#Lag used', 'Number of observation used'])
    for key, val in dftest[4].items():
        dfout['Critical value (%s)'%key]=val
    print(dfout)
    
test_stationary(dif_log)
plt.figure(figsize=(12,7))
edw_avg=train_count_log.ewm(halflife=7, min_periods=0, adjust=True).mean()
train_count_log.plot(label='Log scale')
edw_avg.plot(label='Exponential Decay Weight MA')
# ADCF test
dif_edw=train_count_log-edw_avg
dif_edw.dropna(inplace=True)
test_stationary(dif_edw)
dif_shift=train_count_log-train_count_log.shift()
dif_shift.dropna(inplace=True)
test_stationary(dif_shift)
# decom=seasonal_decompose(train_count_log)
decom=seasonal_decompose(dif_edw)

trend=decom.trend
seasonal=decom.seasonal
residual=decom.resid

fig=plt.figure(figsize=(12,7))
plt.subplot(411)
train_count_log.plot(label='Original')
plt.subplot(412)
trend.plot(label='Trend')
plt.subplot(413)
seasonal.plot(label='Seasonal')
plt.subplot(414)
residual.plot(label='Residual')
fig.tight_layout()

decom_log_data=residual
decom_log_data.dropna(inplace=True)
test_stationary(decom_log_data)
def find_zero_intersection(y):
    # find intersection
    first_line = LineString(np.column_stack((np.arange(len(y)), y)))
    second_line = LineString(np.column_stack((np.arange(len(y)), [0]*len(y))))
    intersection = first_line.intersection(second_line)
    point=list(LineString(intersection).xy[0])
    return (point, [0]*len(point))
# select Exponential Decay Weight Transformation
lag_acf=acf(dif_edw, nlags=20)
lag_pacf=pacf(dif_edw, nlags=20, method='ols')

# plot ACF
fig=plt.figure(figsize=(12,7))
plt.subplot(211)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dif_edw)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(dif_edw)), linestyle='--', color='gray')
# find intersection
x,y=find_zero_intersection(lag_acf)
plt.plot(x,y,'o')
plt.title('Autocorrelation Function') 
print('Q (MA part): ', x[0])

# plot PACF
plt.subplot(212)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(dif_edw)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(dif_edw)), linestyle='--', color='gray')
# find intersection
x,y=find_zero_intersection(lag_pacf)
plt.plot(x,y,'o')
plt.title('Partial Autocorrelation Function') 
print('P (AR part): ', x[0])

fig.tight_layout()
# AR Model
model=ARIMA(train_count_log, order=(2,1,0))
results_AR=model.fit(disp=0)

plt.figure(figsize=(12,7))
dif_edw.plot(label='Exponentian Decay Differentiation')
results_AR.fittedvalues.dropna(inplace=True)
results_AR.fittedvalues.plot(label='Results AR')
df=pd.concat([results_AR.fittedvalues, dif_edw], axis=1).dropna()
plt.title('RSS: %.4f'%sum((df[0]-df['Count'])**2))
# MA Model
model=ARIMA(train_count_log, order=(0,1,2))
results_MA=model.fit(disp=0)

plt.figure(figsize=(12,7))
dif_edw.plot(label='Exponentian Decay Differentiation')
results_MA.fittedvalues.dropna(inplace=True)
results_MA.fittedvalues.plot(label='Results AR')
df=pd.concat([results_MA.fittedvalues, dif_edw], axis=1).dropna()
plt.title('RSS: %.4f'%sum((df[0]-df['Count'])**2))
# ARIMA Model
model=ARIMA(train_count_log, order=(2,1,2))
results_ARIMA=model.fit(disp=0)

plt.figure(figsize=(12,7))
dif_edw.plot(label='Exponentian Decay Differentiation')
results_ARIMA.fittedvalues.dropna(inplace=True)
results_ARIMA.fittedvalues.plot(label='Results AR')
df=pd.concat([results_ARIMA.fittedvalues, dif_edw], axis=1).dropna()
plt.title('RSS: %.4f'%sum((df[0]-df['Count'])**2))
# using AR model
pred_ar_dif=pd.Series(results_AR.fittedvalues, copy=True)
pred_ar_dif_cumsum=pred_ar_dif.cumsum()

pred_ar_log=pd.Series(train_count_log.iloc[0], index=train_count_log.index)
pred_ar_log=pred_ar_log.add(pred_ar_dif_cumsum, fill_value=0)
pred_ar_log.head()

# inverse of log is exp
pred_ar=np.exp(pred_ar_log)
plt.figure(figsize=(12,7))
train.Count.plot(label='Train')
pred_ar.plot(label='Pred')
def validation(order):
    # forecasting for validation
    valid_count_log=list(np.log(valid.Count).values)
    history = list(train_count_log.values)
    model = ARIMA(history, order=order)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=len(valid))
    mse = mean_squared_error(valid_count_log, output[0])
    rmse = np.sqrt(mse)
    print('Test MSE: %.3f' % mse)
    print('Test RMSE: %.3f' % rmse)
    
    fig=plt.figure(figsize=(12,7))
    # reverse transform
    pred=np.exp(output[0])
    pred=pd.Series(pred, index=valid.index)
    valid.Count.plot(label='Valid')
    pred.plot(label='Pred')
    plt.legend(loc='best')
    
    fig=plt.figure(figsize=(12,7))
    train.Count.plot(label='Train')
    valid.Count.plot(label='Valid')
    pred.plot(label='Pred', color='black')

validation((2,1,0))
def arima_predict_hourly(data, arima_order):
    # forecasting for testing (Hourly based forecasting)
    history = data
    model = ARIMA(history, order=arima_order)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=len(test_original))

    submit=test_original.copy()
    submit.index=submit.ID
    submit['Count']=np.exp(output[0])
    submit.drop(['Unnamed: 0','ID','Datetime','year','month','day','hour'], axis=1, inplace=True)
    
    # plot result
    plt.figure(figsize=(12,7))
    train_original.index=train_original.Datetime
    submit.index=test_original.Datetime

    train_original.Count.plot(label='Train')
    submit.Count.plot(label='Pred')
    return submit
# forecasting for testing (Hourly based forecasting)
history = list(np.log(train_original.Count).values)
model = ARIMA(history, order=(2,1,0))
model_fit = model.fit(disp=0)
output = model_fit.forecast(steps=len(test_original))

submit=test_original.copy()
submit.index=submit.ID
submit['Count']=np.exp(output[0])
submit.drop(['Unnamed: 0','ID','Datetime','year','month','day','hour'], axis=1, inplace=True)
# plot result
plt.figure(figsize=(12,7))
train_original.index=train_original.Datetime
submit.index=test_original.Datetime

train_original.Count.plot(label='Train')
submit.Count.plot(label='Pred')
# submission
# submit.to_csv('submit2.csv')
# score 250 (Best Score)
# forecasting for testing (Daily based forecasting)
history = list(np.log(train.Count).values)
model = ARIMA(history, order=(2,1,0))
model_fit = model.fit(disp=0)
output = model_fit.forecast(steps=len(test_df))

test_df['pred']=np.exp(output[0])
train_original['ratio']=train_original['Count']/train_original['Count'].sum() 
temp=train_original.groupby('hour')['ratio'].sum().reset_index()

merge=pd.merge(test_df, test_original, on=('day','month', 'year'), how='left')
merge['hour']=merge.hour_y
merge['ID']=merge['ID_y']
merge=merge.drop(['Unnamed: 0_x','ID_x','year', 'month','hour_x',
                  'Unnamed: 0_y','Datetime','hour_y','ID_y'], axis=1) 
pred=pd.merge(merge, temp, on='hour', how='left')

# convert the ratio to the original scale
pred['Count']=pred['pred']*pred['ratio']*24
plt.figure(figsize=(12,7))
submit=pd.DataFrame(pred.Count.values, columns=['Count'], index=pred.ID)
submit.index=test_original.Datetime
train_original.Count.plot(label='Train')
submit.Count.plot(label='Pred')
submit.index=test_original.ID
# submit.to_csv('submit6.csv')
# score 280
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(arima_order):
    # forecasting for validation
    valid_count_log=list(np.log(valid.Count).values)
    history = list(train_count_log.values)
    model = ARIMA(history, order=arima_order)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=len(valid))
    mse = mean_squared_error(valid_count_log, output[0])
    rmse = np.sqrt(mse)
#     print('Test MSE: %.3f' % mse)
#     print('Test RMSE: %.3f' % rmse)
    return mse
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(p_values, d_values, q_values)
validation((8,1,2))
# forecasting for testing (Hourly based forecasting)
history = list(np.log(train_original.Count).values)
model = ARIMA(history, order=(8,1,2))
model_fit = model.fit(disp=0)
output = model_fit.forecast(steps=len(test_original))

submit=test_original.copy()
submit.index=submit.ID
submit['Count']=np.exp(output[0])
submit.drop(['Unnamed: 0','ID','Datetime','year','month','day','hour'], axis=1, inplace=True)
# plot result
plt.figure(figsize=(12,7))
train_original.index=train_original.Datetime
submit.index=test_original.Datetime

train_original.Count.plot(label='Train')
submit.Count.plot(label='Pred')
# submission
submit.index=test_original.ID
# submit.to_csv('submit5.csv')
# score 260
