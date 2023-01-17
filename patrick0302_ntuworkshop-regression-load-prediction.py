# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import lightgbm as lgb

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import time

import datetime as datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.set_config_file(offline=True)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path_file = r'/kaggle/input/smart-meter-dataset/'
df_metadata_loop = pd.read_csv(os.path.join(path_file, 'metadata_loop.csv'))
df_metadata_loop
df_metadata_uid = pd.read_csv(os.path.join(path_file, 'metadata_uid.csv'))
df_metadata_uid
df_weather = pd.read_csv(os.path.join(path_file, 'weather.csv'))
df_weather['ObsTime'] = pd.to_datetime(df_weather['ObsTime'])
df_weather
df_holiday_encode = pd.read_csv(os.path.join(path_file, 'holiday_encode.csv'))
df_holiday_encode['date'] = pd.to_datetime(df_holiday_encode['date'])
df_holiday_encode
df_powerMeter_pivot_output = pd.read_csv(os.path.join(path_file, 'powerMeter.csv'))
df_powerMeter_pivot_output['日期時間'] = pd.to_datetime(df_powerMeter_pivot_output['日期時間'])
df_powerMeter_pivot_output = df_powerMeter_pivot_output.set_index('日期時間')
df_powerMeter_pivot_output
building_type = '行政單位'

df_metadata = df_metadata_loop.merge(df_metadata_uid, on='uid')
list_powerMeter = df_metadata[df_metadata['buildType1C']==building_type]['迴路編號'].to_list()
df_powerMeter_pivot_output = df_powerMeter_pivot_output.loc[:, df_powerMeter_pivot_output.columns.str.contains('|'.join(list_powerMeter))]
df_powerMeter_pivot_output.columns
# Normalize the energy data and take average of all meters' trends
df_powerMeter_pivot_output = (df_powerMeter_pivot_output-df_powerMeter_pivot_output.mean())/df_powerMeter_pivot_output.std()
df_powerMeter_pivot_output[building_type + '_mean'] = df_powerMeter_pivot_output.mean(axis=1) + 1

meter_name = building_type + '_mean'

# Prepare data for modeling
df_temp = df_powerMeter_pivot_output.loc[:, meter_name].reset_index().copy()
df_temp = df_temp.dropna()

# Add timestamp features
df_temp['weekday'] = df_temp['日期時間'].dt.weekday
df_temp['hour'] = df_temp['日期時間'].dt.hour
df_temp['date'] =pd.to_datetime(df_temp['日期時間'].dt.date)

df_temp = df_temp.set_index('日期時間').drop(['date'],axis=1)

df_temp = df_temp.rename(columns={meter_name:'elec_meas'})

df_temp
# Weekly profiles of building energy
df_plot = df_temp.copy()
df_plot['elec_meas'].iplot()
df_plot['date'] = pd.to_datetime(df_plot.index.date)
df_plot.pivot_table(columns=['weekday','hour'], index='date', values='elec_meas').T.plot(figsize=(15,5),color='black',alpha=0.1,legend=False)
traindata = df_temp.loc['2016'].copy()
testdata = df_temp.loc['2017'].copy()

train_labels = traindata['elec_meas']
test_labels = testdata['elec_meas']

train_features = traindata.drop('elec_meas', axis=1)
test_features = testdata.drop('elec_meas', axis=1) 

LGB_model = lgb.LGBMRegressor()
LGB_model.fit(train_features, train_labels)

testdata['elec_pred'] = LGB_model.predict(test_features)

df_temp.loc['2017', 'elec_pred'] = testdata['elec_pred']

# Calculate the absolute errors
errors = abs(testdata['elec_pred'] - test_labels)

# Calculate mean absolute percentage error (MAPE) and add to list
MAPE = 100 * np.mean((errors / test_labels))
NMBE = 100 * (sum(testdata.dropna()['elec_meas'] - testdata.dropna()['elec_pred']) / (testdata.dropna()['elec_meas'].count() * np.mean(testdata.dropna()['elec_meas'])))
CVRSME = 100 * ((sum((testdata.dropna()['elec_meas'] - testdata.dropna()['elec_pred'])**2) / (testdata.dropna()['elec_meas'].count()-1))**(0.5)) / np.mean(testdata.dropna()['elec_meas'])
RSQUARED = r2_score(testdata.dropna()['elec_meas'], testdata.dropna()['elec_pred'])

print("MAPE: "+str(round(MAPE,2)))
print("NMBE: "+str(round(NMBE,2)))
print("CVRSME: "+str(round(CVRSME,2)))
print("R SQUARED: "+str(round(RSQUARED,2)))

testdata[['elec_meas', 'elec_pred']].iplot()
df_temp = df_powerMeter_pivot_output.loc[:, meter_name].reset_index().copy()
df_temp = df_temp.dropna()

# Add timestamp features
df_temp['weekday'] = df_temp['日期時間'].dt.weekday
df_temp['hour'] = df_temp['日期時間'].dt.hour
df_temp['date'] =pd.to_datetime(df_temp['日期時間'].dt.date)

# Add weather features
df_temp = df_temp.merge(df_weather[['ObsTime', 'Temperature']], left_on='日期時間', right_on='ObsTime')

df_temp = df_temp.set_index('日期時間').drop(['ObsTime','date'],axis=1)

df_temp = df_temp.rename(columns={meter_name:'elec_meas'})

df_temp
# Scatter plot for energy consumptions and outdoor temperature
plt.figure(figsize=(10,10))
df_plot = df_temp.copy()
df_plot = df_plot[df_plot['elec_meas']<3]
df_plot['weekday/weekend'] = 'weekday'
df_plot.loc[df_plot['weekday']>4, 'weekday/weekend'] ='weekend'
df_plot
ax = sns.relplot(x="Temperature", y="elec_meas", col="weekday/weekend", hue='hour',
                 kind="scatter", data=df_plot, alpha=0.1)
traindata = df_temp.loc['2016'].copy()
testdata = df_temp.loc['2017'].copy()

train_labels = traindata['elec_meas']
test_labels = testdata['elec_meas']

train_features = traindata.drop('elec_meas', axis=1)
test_features = testdata.drop('elec_meas', axis=1) 

LGB_model = lgb.LGBMRegressor()
LGB_model.fit(train_features, train_labels)

testdata['elec_pred'] = LGB_model.predict(test_features)

# Use the forest's predict method on the train data
df_temp.loc['2017', 'elec_pred'] = testdata['elec_pred']

# Calculate the absolute errors
errors = abs(testdata['elec_pred'] - test_labels)
# Print out the mean absolute error (mae)

# Calculate mean absolute percentage error (MAPE) and add to list
MAPE = 100 * np.mean((errors / test_labels))
NMBE = 100 * (sum(testdata.dropna()['elec_meas'] - testdata.dropna()['elec_pred']) / (testdata.dropna()['elec_meas'].count() * np.mean(testdata.dropna()['elec_meas'])))
CVRSME = 100 * ((sum((testdata.dropna()['elec_meas'] - testdata.dropna()['elec_pred'])**2) / (testdata.dropna()['elec_meas'].count()-1))**(0.5)) / np.mean(testdata.dropna()['elec_meas'])
RSQUARED = r2_score(testdata.dropna()['elec_meas'], testdata.dropna()['elec_pred'])

print("MAPE: "+str(round(MAPE,2)))
print("NMBE: "+str(round(NMBE,2)))
print("CVRSME: "+str(round(CVRSME,2)))
print("R SQUARED: "+str(round(RSQUARED,2)))

testdata[['elec_meas', 'elec_pred']].iplot()
df_temp = df_powerMeter_pivot_output.loc[:, meter_name].reset_index().copy()
df_temp = df_temp.dropna()

# Add timestamp features
df_temp['weekday'] = df_temp['日期時間'].dt.weekday
df_temp['hour'] = df_temp['日期時間'].dt.hour
df_temp['date'] =pd.to_datetime(df_temp['日期時間'].dt.date)

# Add weather features
df_temp = df_temp.merge(df_weather[['ObsTime', 'Temperature']], left_on='日期時間', right_on='ObsTime')

# Add holiday features
df_temp = df_temp.merge(df_holiday_encode, on='date')

df_temp = df_temp.set_index('日期時間').drop(['ObsTime','date'],axis=1)

df_temp = df_temp.rename(columns={meter_name:'elec_meas'})

df_temp
traindata = df_temp.loc['2016'].copy()
testdata = df_temp.loc['2017'].copy()

train_labels = traindata['elec_meas']
test_labels = testdata['elec_meas']

train_features = traindata.drop('elec_meas', axis=1)
test_features = testdata.drop('elec_meas', axis=1) 

LGB_model = lgb.LGBMRegressor()
LGB_model.fit(train_features, train_labels)

testdata['elec_pred'] = LGB_model.predict(test_features)

# Use the forest's predict method on the train data
df_temp.loc['2017', 'elec_pred'] = testdata['elec_pred']

# Calculate the absolute errors
errors = abs(testdata['elec_pred'] - test_labels)
# Print out the mean absolute error (mae)

# Calculate mean absolute percentage error (MAPE) and add to list
MAPE = 100 * np.mean((errors / test_labels))
NMBE = 100 * (sum(testdata.dropna()['elec_meas'] - testdata.dropna()['elec_pred']) / (testdata.dropna()['elec_meas'].count() * np.mean(testdata.dropna()['elec_meas'])))
CVRSME = 100 * ((sum((testdata.dropna()['elec_meas'] - testdata.dropna()['elec_pred'])**2) / (testdata.dropna()['elec_meas'].count()-1))**(0.5)) / np.mean(testdata.dropna()['elec_meas'])
RSQUARED = r2_score(testdata.dropna()['elec_meas'], testdata.dropna()['elec_pred'])

print("MAPE: "+str(round(MAPE,2)))
print("NMBE: "+str(round(NMBE,2)))
print("CVRSME: "+str(round(CVRSME,2)))
print("R SQUARED: "+str(round(RSQUARED,2)))

testdata[['elec_meas', 'elec_pred']].iplot()
testdata[['elec_meas', 'elec_pred']].resample('D').mean().iplot()
testdata[['elec_pred', 'elec_meas']].resample('M').mean().iplot(kind='bar')
(testdata[['elec_pred', 'elec_meas']].resample('Y').mean()+1).iplot(kind='bar')

