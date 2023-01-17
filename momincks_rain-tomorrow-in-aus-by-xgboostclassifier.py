import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')
df = df.drop('RISK_MM',axis=1)
print(df.isna().sum())
print('Total',len(df.columns),'features\n',df.dtypes)
df['Month'] = df['Date'].str.slice(start=5,stop=7) # Get Month from Date
df['Date'] = pd.to_datetime(df['Date'],format='%Y/%m/%d',errors='ignore')
df_dateplot = df.iloc[-900:,:]
plt.figure(figsize=[20,3])
plt.plot(df_dateplot['Date'],df_dateplot['MinTemp'],color='blue')
plt.plot(df_dateplot['Date'],df_dateplot['MaxTemp'],color='red')
plt.fill_between(df_dateplot['Date'],df_dateplot['MinTemp'],df_dateplot['MaxTemp'], facecolor = '#EBF78F')
plt.legend()
plt.show()
df['Season_Q1'] = (df['Month']=='01') | (df['Month']=='02') | (df['Month']=='03')
df['Season_Q2'] = (df['Month']=='04') | (df['Month']=='05') | (df['Month']=='06')
df['Season_Q3'] = (df['Month']=='07') | (df['Month']=='08') | (df['Month']=='09')
df['Year_FirstHalf'] = df['Season_Q1'] | df['Season_Q2']
df['NoRain'] = (df['Rainfall'] == 0)
df['Temp_MinMax'] = df['MaxTemp'] - df['MinTemp']
df['Temp_delta'] = df['Temp3pm'] - df['Temp9am']
df['Humidity_delta'] = df['Humidity3pm'] - df['Humidity9am']
df['WindSpeed_delta'] = df['WindSpeed3pm'] - df['WindSpeed9am']
df['Cloud_delta'] = df['Cloud3pm'] - df['Cloud9am']
df['Pressure_delta'] = df['Pressure3pm'] - df['Pressure9am']
df['NoSunshine'] = (df['Sunshine'] == 0)
df['HighSunshine'] = (df['Sunshine'] >= df['Sunshine'].median())
df['LowHumidity3pm'] = (df['Humidity3pm'] <= df['Humidity3pm'].median())
df['LowCloud3pm'] = (df['Cloud3pm'] <= df['Cloud3pm'].mean())
print(df.dtypes)
df_hist = df.select_dtypes(exclude = ['bool','object'])
df_hist.hist(figsize = [15,15],bins = 50)
plt.show()
for i in df_hist.columns:
    df[[i]] = preprocessing.StandardScaler().fit_transform(df[[i]])
df['Rainfall'] = df['Rainfall'].apply(lambda x: np.log(x) if x>0 else x)
df['Evaporation'] = df['Evaporation'].apply(lambda x: np.log(x) if x>0 else x)
features_to_transform = ['Evaporation','Humidity9am','Sunshine','Rainfall']
for i in features_to_transform:
    df[[i]] = preprocessing.QuantileTransformer(n_quantiles=100,output_distribution='normal',subsample=len(df)).fit_transform(df[[i]])
df_hist = df[features_to_transform]
df_hist.hist(figsize = [15,15],bins = 50)
plt.show()
df['Humidity9am_transformer'] = (df['Humidity9am']>4)
df['Sunshine_transformer'] = (df['Sunshine']>-4)
features_to_drop = ['Date']
df = df.drop(features_to_drop,axis=1)
remained_categorial_data = ['Month','WindGustDir','WindDir9am','WindDir3pm','RainToday','RainTomorrow','Location']
df_onehotted = pd.get_dummies(df,columns=remained_categorial_data,drop_first=True)
asc = df_onehotted.corrwith(df_onehotted['RainTomorrow_Yes']).sort_values(ascending=True)[:10]
desc = df_onehotted.corrwith(df_onehotted['RainTomorrow_Yes']).sort_values(ascending=False)[1:11]
print(desc)
print(asc)
x_train, x_test, y_train, y_test = train_test_split(df_onehotted.drop(['RainTomorrow_Yes'],axis=1),df_onehotted['RainTomorrow_Yes'],test_size = 0.2, random_state = 0)
%%time

import xgboost as xgb

xgb = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.9,
              colsample_bynode=0.9, colsample_bytree=0.5, gamma=0,
              grow_policy='lossguide', learning_rate=0.4, max_bin=512,
              max_delta_step=0, max_depth=8, min_child_weight=0.8, missing=None,
              n_estimators=100, n_jobs=1, nthread=None, num_parallel_tree=9,
              objective='binary:hinge', random_state=0, reg_alpha=2,
              reg_lambda=3, sampling_method='uniform', scale_pos_weight=1,
              seed=None, silent=None, subsample=0.8, tree_method='hist',
              verbosity=1)

xgb.fit(x_train, y_train)
pred = xgb.predict(x_test)
print('acc',metrics.accuracy_score(y_test,pred))
print('f1',metrics.f1_score(y_test,pred))
print('matrix',metrics.confusion_matrix(y_test,pred))
gain = xgb.get_booster().get_score(importance_type='gain')
gain = pd.DataFrame.from_dict(gain,orient='index',columns=['gain']).sort_values(by=['gain'],ascending=False)[:10]
print(gain.to_string())
cover = xgb.get_booster().get_score(importance_type='cover')
cover = pd.DataFrame.from_dict(cover,orient='index',columns=['cover']).sort_values(by=['cover'],ascending=False)[:10]
print(cover.to_string())