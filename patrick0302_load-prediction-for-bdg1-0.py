# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport



import matplotlib.pyplot as plt

from matplotlib import dates as md

import seaborn as sns

import plotly.graph_objs as go

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.set_config_file(offline=True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import lightgbm as lgb



from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error



import statsmodels.api as sm

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt
df_meta = pd.read_csv('/kaggle/input/building-data-genome-project-v1/meta_open.csv')

df_meta
df_meta[df_meta['newweatherfilename']=='weather2.csv']
df_powerMeter = pd.read_csv('/kaggle/input/building-data-genome-project-v1/temp_open_utc_complete.csv', index_col='timestamp', parse_dates=True)

df_powerMeter.index = df_powerMeter.index.tz_localize(None)

df_powerMeter = df_powerMeter/df_meta.set_index('uid').loc[df_powerMeter.columns, 'sqm']

df_powerMeter
list_bldg_site2 = df_meta.loc[df_meta['newweatherfilename']=='weather2.csv', 'uid'].to_list()
df_powerMeter_site2 =  df_powerMeter[list_bldg_site2].dropna(how='all')

df_powerMeter_site2
df_weather2 = pd.read_csv('/kaggle/input/building-data-genome-project-v1/weather2.csv', index_col='timestamp', parse_dates=True)

df_weather2 = df_weather2.select_dtypes(['int', 'float'])



for col in df_weather2.columns:

    df_weather2.loc[df_weather2[col]<-100, col] = np.nan



df_weather2 = df_weather2.reset_index().drop_duplicates(subset=['timestamp'])



df_weather2 = df_weather2.set_index('timestamp').resample('1H').mean()

#df_weather2 = df_weather2.interpolate('cubicspline')



df_weather2
df_weather2.iplot()
df_weather2['TemperatureC_movingAvg_3hr'] = df_weather2['TemperatureC'].rolling(3).mean()

df_weather2['TemperatureC_movingAvg_6hr'] = df_weather2['TemperatureC'].rolling(6).mean()

df_weather2['TemperatureC_movingAvg_12hr'] = df_weather2['TemperatureC'].rolling(12).mean()

df_weather2['TemperatureC_movingAvg_24hr'] = df_weather2['TemperatureC'].rolling(24).mean()

df_weather2.loc[:, df_weather2.columns.str.contains('TemperatureC')].iplot()
df_schedule2 = pd.read_csv('/kaggle/input/building-data-genome-project-v1/schedule2.csv', header=None)

df_schedule2 = df_schedule2.rename(columns={0:'date',1:'date_type'})

df_schedule2['date'] = pd.to_datetime(df_schedule2['date'])

df_schedule2
df_schedule2_encode = df_schedule2.copy()

df_schedule2_encode['date_type'] = LabelEncoder().fit_transform(df_schedule2_encode['date_type'])

df_schedule2_encode
df_schedule2_encode.set_index('date').iplot(kind='bar')
df_holiday = pd.read_html('https://www.timeanddate.com/holidays/us/2015')[0]

df_holiday.columns = df_holiday.columns.get_level_values(0)

df_holiday = df_holiday.loc[df_holiday['Date'].str.len()<100]

df_holiday = df_holiday[['Date', 'Name', 'Type']]

df_holiday['Date'] = '2015 ' + df_holiday['Date']

df_holiday['Date'] = pd.to_datetime(df_holiday['Date'])

df_holiday = df_holiday.rename(columns={'Date':'date'})



df_holiday = df_holiday.drop_duplicates(subset=['date'])

df_holiday = df_holiday.set_index('date').asfreq('D')



df_holiday.loc[df_holiday.index.weekday>=5, 'Name'] = 'weekend'

df_holiday.loc[df_holiday.index.weekday>=5, 'Type'] = 'weekend'



df_holiday.columns = 'holiday_' + df_holiday.columns



df_holiday
df_holiday_encode = df_holiday.copy()

df_holiday_encode[['holiday_Name', 'holiday_Type']] = df_holiday_encode[['holiday_Name', 'holiday_Type']].astype('str').apply(LabelEncoder().fit_transform)

df_holiday_encode
df_holiday_encode.iplot(kind='bar')
name_meter = 'Office_Caleb'
# Prepare data for modeling

df_temp = df_powerMeter_site2[[name_meter]].copy()

df_temp = df_temp.dropna()



# Add timestamp features

df_temp['weekday'] = df_temp.index.weekday

df_temp['hour'] = df_temp.index.hour

df_temp['date'] = df_temp.index.date



df_temp = df_temp.rename(columns={name_meter: 'load_meas'})



df_temp
# Weekly profiles of building energy

df_plot = df_temp.copy()

df_plot['date'] = pd.to_datetime(df_plot.index.date)

df_plot.pivot_table(columns=['weekday','hour'], index='date', values='load_meas').T.plot(figsize=(15,5),color='black',alpha=0.1,legend=False)
traindata = df_temp.loc[df_temp.index.month.isin([1,2,3,5,6,7,9,10,11])].copy()

testdata = df_temp.loc[df_temp.index.month.isin([4,8,12])].copy()



train_labels = traindata['load_meas']

test_labels = testdata['load_meas']



train_features = traindata.drop(['load_meas', 'date'], axis=1)

test_features = testdata.drop(['load_meas', 'date'], axis=1)



LGB_model = lgb.LGBMRegressor()

LGB_model.fit(train_features, train_labels)



testdata['load_pred'] = LGB_model.predict(test_features)

df_temp.loc[testdata.index, 'load_pred'] = testdata['load_pred']



# Calculate the absolute errors

errors = abs(testdata['load_pred'] - test_labels)



RSQUARED = r2_score(testdata.dropna()['load_meas'], testdata.dropna()['load_pred'])



print("R SQUARED: "+str(round(RSQUARED,3)))

testdata[['load_meas', 'load_pred']].iplot()
# Prepare data for modeling

df_temp = df_powerMeter_site2[[name_meter]].copy()

df_temp = df_temp.dropna()



# Add timestamp features

df_temp['weekday'] = df_temp.index.weekday

df_temp['hour'] = df_temp.index.hour

df_temp['date'] = df_temp.index.date



# Add weather features

df_temp = df_temp.rename(columns={name_meter: 'load_meas'})

df_temp = df_temp.merge(df_weather2.loc[:, df_weather2.columns.str.contains('TemperatureC')], left_index=True, right_index=True)



df_temp
# Scatter plot for energy consumptions and outdoor temperature

plt.figure(figsize=(10,10))

df_plot = df_temp.copy()

df_plot = df_plot.resample('D').mean()

df_plot['weekday/weekend'] = 'weekday'

df_plot.loc[df_plot['weekday']>4, 'weekday/weekend'] ='weekend'



ax = sns.relplot(x="TemperatureC", y="load_meas", col="weekday/weekend",

                 kind="scatter", data=df_plot, alpha=0.8)
traindata = df_temp.loc[df_temp.index.month.isin([1,2,3,5,6,7,9,10,11])].dropna().copy()

testdata = df_temp.loc[df_temp.index.month.isin([4,8,12])].copy()



train_labels = traindata['load_meas']

test_labels = testdata['load_meas']



train_features = traindata.drop(['load_meas', 'date'], axis=1)

test_features = testdata.drop(['load_meas', 'date'], axis=1)



LGB_model = lgb.LGBMRegressor()

LGB_model.fit(train_features, train_labels)



testdata['load_pred'] = LGB_model.predict(test_features)

df_temp.loc[testdata.index, 'load_pred'] = testdata['load_pred']



# Calculate the absolute errors

errors = abs(testdata['load_pred'] - test_labels)



RSQUARED = r2_score(testdata.dropna()['load_meas'], testdata.dropna()['load_pred'])



print("R SQUARED: "+str(round(RSQUARED,3)))

testdata[['load_meas', 'load_pred']].iplot()
# Prepare data for modeling

df_temp = df_powerMeter_site2[[name_meter]].copy()

df_temp = df_temp.dropna()



# Add timestamp features

df_temp['weekday'] = df_temp.index.weekday

df_temp['hour'] = df_temp.index.hour

df_temp['date'] = pd.to_datetime(df_temp.index.date)



# Add weather features

df_temp = df_temp.rename(columns={name_meter: 'load_meas'})

df_temp = df_temp.merge(df_weather2.loc[:, df_weather2.columns.str.contains('TemperatureC')], left_index=True, right_index=True)



# Add holiday features

idx_df = df_temp.index.copy()

df_temp = df_temp.merge(df_holiday_encode[['holiday_Type']].reset_index(), on='date')

df_temp.index = idx_df



df_temp
traindata = df_temp.loc[df_temp.index.month.isin([1,2,3,5,6,7,9,10,11])].dropna().copy()

testdata = df_temp.loc[df_temp.index.month.isin([4,8,12])].copy()



train_labels = traindata['load_meas']

test_labels = testdata['load_meas']



train_features = traindata.drop(['load_meas', 'date'], axis=1)

test_features = testdata.drop(['load_meas', 'date'], axis=1)



LGB_model = lgb.LGBMRegressor()

LGB_model.fit(train_features, train_labels)



testdata['load_pred'] = LGB_model.predict(test_features)

df_temp.loc[testdata.index, 'load_pred'] = testdata['load_pred']



# Calculate the absolute errors

errors = abs(testdata['load_pred'] - test_labels)



RSQUARED = r2_score(testdata.dropna()['load_meas'], testdata.dropna()['load_pred'])



print("R SQUARED: "+str(round(RSQUARED,3)))

testdata[['load_meas', 'load_pred']].iplot()
# Prepare data for modeling

df_temp = df_powerMeter_site2[[name_meter]].copy()

df_temp = df_temp.dropna()



# Add timestamp features

df_temp['weekday'] = df_temp.index.weekday

df_temp['hour'] = df_temp.index.hour

df_temp['date'] = pd.to_datetime(df_temp.index.date)



# Add weather features

df_temp = df_temp.rename(columns={name_meter: 'load_meas'})

df_temp = df_temp.merge(df_weather2.loc[:, df_weather2.columns.str.contains('TemperatureC')], left_index=True, right_index=True)



# Add schedule features

idx_df = df_temp.index.copy()

df_temp = df_temp.merge(df_schedule2_encode, on='date')

df_temp.index = idx_df



df_temp
traindata = df_temp.loc[df_temp.index.month.isin([1,2,3,5,6,7,9,10,11])].dropna().copy()

testdata = df_temp.loc[df_temp.index.month.isin([4,8,12])].copy()



train_labels = traindata['load_meas']

test_labels = testdata['load_meas']



train_features = traindata.drop(['load_meas', 'date'], axis=1)

test_features = testdata.drop(['load_meas', 'date'], axis=1)



LGB_model = lgb.LGBMRegressor()

LGB_model.fit(train_features, train_labels)



testdata['load_pred'] = LGB_model.predict(test_features)

df_temp.loc[testdata.index, 'load_pred'] = testdata['load_pred']



# Calculate the absolute errors

errors = abs(testdata['load_pred'] - test_labels)



RSQUARED = r2_score(testdata.dropna()['load_meas'], testdata.dropna()['load_pred'])



print("R SQUARED: "+str(round(RSQUARED,3)))

testdata[['load_meas', 'load_pred']].iplot()
# Prepare data for modeling

df_temp = df_powerMeter_site2[[name_meter]].copy()

df_temp = df_temp.dropna()



# Add timestamp features

df_temp['weekday'] = df_temp.index.weekday

df_temp['hour'] = df_temp.index.hour

df_temp['date'] = pd.to_datetime(df_temp.index.date)



# Add weather features

df_temp = df_temp.rename(columns={name_meter: 'load_meas'})

df_temp = df_temp.merge(df_weather2.loc[:, df_weather2.columns.str.contains('TemperatureC')], left_index=True, right_index=True)



# Add holiday features

idx_df = df_temp.index.copy()

df_temp = df_temp.merge(df_schedule2_encode, on='date')

df_temp.index = idx_df



# Add lag features

df_temp['load_shift_24hrs'] = df_temp['load_meas'].shift(24)



df_temp
fig = plt.figure(figsize=(12,8))



ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(df_temp['load_meas'], lags=24*7, ax=ax1)

ax1.xaxis.set_ticks_position('bottom')

fig.tight_layout();



ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(df_temp['load_meas'], lags=24*7, ax=ax2)

ax2.xaxis.set_ticks_position('bottom')

fig.tight_layout();
traindata = df_temp.loc[df_temp.index.month.isin([1,2,3,5,6,7,9,10,11])].dropna().copy()

testdata = df_temp.loc[df_temp.index.month.isin([4,8,12])].copy()



train_labels = traindata['load_meas']

test_labels = testdata['load_meas']



train_features = traindata.drop(['load_meas', 'date'], axis=1)

test_features = testdata.drop(['load_meas', 'date'], axis=1)



LGB_model = lgb.LGBMRegressor()

LGB_model.fit(train_features, train_labels)



testdata['load_pred'] = LGB_model.predict(test_features)

df_temp.loc[testdata.index, 'load_pred'] = testdata['load_pred']



# Calculate the absolute errors

errors = abs(testdata['load_pred'] - test_labels)



RSQUARED = r2_score(testdata.dropna()['load_meas'], testdata.dropna()['load_pred'])



print("R SQUARED: "+str(round(RSQUARED,3)))

testdata[['load_meas', 'load_pred']].iplot()
df_powerMeter_unnormalized = pd.read_csv('/kaggle/input/building-data-genome-project-v1/temp_open_utc_complete.csv', index_col='timestamp', parse_dates=True)

df_powerMeter_unnormalized.index = df_powerMeter.index.tz_localize(None)
df_model_prediction = pd.DataFrame()
for name_meter in list_bldg_site2:

    print(name_meter)



    # Prepare data for modeling

    df_temp = df_powerMeter_unnormalized[[name_meter]].copy()

    df_temp = df_temp.dropna()



    # Add timestamp features

    df_temp['weekday'] = df_temp.index.weekday

    df_temp['hour'] = df_temp.index.hour

    df_temp['date'] = pd.to_datetime(df_temp.index.date)



    # Add weather features

    df_temp = df_temp.rename(columns={name_meter: 'load_meas'})

    df_temp = df_temp.merge(df_weather2.loc[:, df_weather2.columns.str.contains('TemperatureC')], left_index=True, right_index=True)



    # Add holiday features

    idx_df = df_temp.index.copy()

    df_temp = df_temp.merge(df_schedule2_encode, on='date')

    df_temp.index = idx_df



    # Add lag features

    df_temp['load_shift_24hrs'] = df_temp['load_meas'].shift(24)



    # Split data for train and test

    traindata = df_temp.loc[df_temp.index.month.isin([1,2,3,5,6,7,9,10,11])].dropna().copy()

    testdata = df_temp.loc[df_temp.index.month.isin([4,8,12])].copy()



    train_labels = traindata['load_meas']

    test_labels = testdata['load_meas']



    train_features = traindata.drop(['load_meas', 'date'], axis=1)

    test_features = testdata.drop(['load_meas', 'date'], axis=1)



    LGB_model = lgb.LGBMRegressor()

    LGB_model.fit(train_features, train_labels)



    testdata['load_pred'] = LGB_model.predict(test_features)

    df_temp.loc[testdata.index, 'load_pred'] = testdata['load_pred']



    # Calculate the absolute errors

    errors = abs(testdata['load_pred'] - test_labels)



    RSQUARED = r2_score(testdata.dropna()['load_meas'], testdata.dropna()['load_pred'])

    MAPE = errors/test_labels

    MAPE = MAPE.loc[MAPE!=np.inf]

    MAPE = MAPE.loc[MAPE!=-np.inf]

    MAPE = MAPE.dropna().mean()*100



    print("R SQUARED: "+str(round(RSQUARED,3)))

    print("MAPE: "+str(round(MAPE,1))+'%')

    testdata[['load_meas', 'load_pred']].reset_index(drop=True).plot(figsize=(15,3), title=name_meter);plt.show()



    testdata['uid'] = name_meter

    testdata['RSQUARED'] = RSQUARED

    testdata['MAPE'] = MAPE



    df_model_prediction = pd.concat([df_model_prediction, testdata[['load_meas', 'load_pred', 'uid','RSQUARED','MAPE']].reset_index()], ignore_index=True, axis=0)
df_model_prediction
df_model_prediction.to_pickle('df_model_prediction.pickle.gz', compression='gzip')