import pandas as pd
# prophet by Facebook
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer

import warnings; warnings.simplefilter('ignore')

# Load data
df = pd.read_csv('../input/120001_PH1.csv')
df.head()
df['ds'] = pd.to_datetime(df['ds'])
#df.index = df['ds']
#del df['ds']
df.tail()


df.plot(x='ds',y='y',figsize=(20, 6))
maxKM = 1200.0
df.y.loc[df.y> maxKM]=maxKM
df.plot(x='ds',y='y',figsize=(20, 6))
count_total_days = df.size
print("count_total_days",count_total_days)

count_zero_days = df[df.y==0].count()
print("count_zero_days",count_zero_days)

count_null_days = df['y'].isna().sum()
print("count_null_days",count_null_days)

perc = count_null_days / count_total_days
print("% of missing days",perc)
forecast_horizon = 30 #days to forcast
train_df = df[:-forecast_horizon]
test_df =  df[-forecast_horizon:]
print("train",train_df.tail())
print("test",test_df.head())
def forecast_by_Prophet(train):
    m = Prophet()
    m.fit(train)
    # Python
    future = m.make_future_dataframe(periods=forecast_horizon)
    forecast = m.predict(future)
    forecast= forecast[['ds', 'yhat']]
    forecast= forecast[-forecast_horizon:] 
    return forecast
forecast = forecast_by_Prophet(train_df)
error_missingData =  mean_absolute_error(test_df['y'], forecast['yhat'])

print("MAE with missing data", error_missingData)
comparison = test_df 
comparison['NoFilling'] = forecast.yhat
comparison.plot(title="comparison",x='ds',figsize=(20, 6))
count_null_days = train_df['y'].isna().sum()
print("count_null_days in train_df",count_null_days)

train_forwardfill = train_df.fillna(method='ffill')

count_null_days = train_forwardfill['y'].isna().sum()
print("count_null_days in train_ffill",count_null_days)

forecast = forecast_by_Prophet(train_forwardfill)
error_ffill = mean_absolute_error(test_df['y'], forecast['yhat'])
print("MAE with forwardfill missing data", error_ffill)
 
comparison['FFILL'] = forecast.yhat
comparison.plot(title="comparison",x='ds',figsize=(20, 6))
train_backwardfill = train_df.fillna(method='bfill')
forecast = forecast_by_Prophet(train_backwardfill)
error_bfill = mean_absolute_error(test_df['y'], forecast['yhat'])
print("MAE with backwardfill missing data", error_bfill)

comparison['BFILL'] = forecast.yhat
comparison.plot(title="comparison",x='ds',figsize=(20, 6))
# other strategies: “median”, most_frequent, constant
my_imputer = Imputer(strategy='mean')
y= train_df['y'].values
y = y.reshape(-1, 1)
y_imputed = my_imputer.fit_transform(y)
train_imputed = train_df
train_imputed= train_imputed.drop('y',1)
train_imputed['y']= y_imputed
forecast = forecast_by_Prophet(train_imputed)
error_imputed_mean = mean_absolute_error(test_df['y'], forecast['yhat'])
print("MAE with imputed Mean", error_imputed_mean)


comparison['MEAN'] = forecast.yhat
comparison.plot(title="comparison",x='ds',figsize=(20, 6))
my_imputer = Imputer(strategy='median')
y= train_df['y'].values
y = y.reshape(-1, 1)
y_imputed = my_imputer.fit_transform(y)
train_imputed = train_df
train_imputed= train_imputed.drop('y',1)
train_imputed['y']= y_imputed
forecast = forecast_by_Prophet(train_imputed)
error_imputed_median = mean_absolute_error(test_df['y'], forecast['yhat'])
print("MAE with imputed median", error_imputed_median)
comparison['MEDIAN'] = forecast.yhat
comparison.plot(title="comparison",x='ds',figsize=(20, 6))
my_imputer = Imputer(strategy='most_frequent')
y= train_df['y'].values
y = y.reshape(-1, 1)
y_imputed = my_imputer.fit_transform(y)
train_imputed = train_df
train_imputed= train_imputed.drop('y',1)
train_imputed['y']= y_imputed
forecast = forecast_by_Prophet(train_imputed)
error_imputed_most_frequent = mean_absolute_error(test_df['y'], forecast['yhat'])
print("MAE with imputed most_frequent", error_imputed_most_frequent)
comparison['MOST_FREQUENT'] = forecast.yhat
comparison.plot(title="comparison",x='ds',figsize=(20, 6))
errors = [error_missingData, error_ffill,error_bfill, error_imputed_mean, error_imputed_median, error_imputed_most_frequent]
methods = ['NA', 'ffill', 'bfill','mean','median', 'most_frequent']
df = pd.DataFrame({'MAE': errors, 'Filling Method': methods}, index=methods)
ax = df.plot.bar(rot=0, figsize=(10, 7))