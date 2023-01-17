!pip install astral
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb

from pandas_profiling import ProfileReport

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_gen = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_sensor = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_gen.info()
profile_gen = ProfileReport(df_gen)
profile_gen.to_widgets()
profile_gen.to_notebook_iframe()
from datetime import datetime

df_gen['DATE_TIME'] = df_gen['DATE_TIME'].apply(lambda x: datetime.strptime(x,'%d-%m-%Y %H:%M'))
df_gen['DATE_TIME'].max()
df_gen['DATE_TIME'].min()
df_gen.info()
# df_gen['MONTH'] = df_gen['DATE_TIME'].dt.month

# df_gen['DAY'] = df_gen['DATE_TIME'].dt.day

df_gen['HOUR'] = df_gen['DATE_TIME'].dt.hour

# df_gen['MINUTE'] = df_gen['DATE_TIME'].dt.minute

df_gen['DATE'] = df_gen['DATE_TIME'].dt.date

df_gen['TIME'] = df_gen['DATE_TIME'].dt.time
df_gen.head()
# Remove PLANT_ID due to it has only 1 distinct value

df_gen = df_gen.drop('PLANT_ID', axis = 1)
df_gen.head()
df_gen['SOURCE_KEY'].value_counts()
print('There are {} Inverters'.format(len(df_gen['SOURCE_KEY'].value_counts())))
# Make it easier to read

temp_dict = {}
for index, source in enumerate(set(df_gen['SOURCE_KEY'])):
    temp_dict[source] = index

temp_dict
df_gen['SOURCE_KEY'] = df_gen['SOURCE_KEY'].map(temp_dict)

df_gen.head()
# Drop due to high correlation with AC_POWER

df_gen = df_gen.drop('DC_POWER', axis = 1)
df_gen.head()
plt.figure(figsize=(20, 8))

sns.lineplot(data=df_gen, x="DATE_TIME", y='AC_POWER', hue='SOURCE_KEY')
# Drop due to TOTAL_YIELD we want to predict is accumulated from DAILY_YIELD

df_gen = df_gen.drop('DAILY_YIELD', axis = 1)
df_gen.head()
import datetime
from astral import LocationInfo
from astral.sun import sun

def light_during_day(date):
    
    # Lat, Long @ India 
    latitude = 78.9629
    longitude = 20.5937
    
    city = LocationInfo("India", latitude, longitude)
    
    year = date.year
    month = date.month
    day = date.day
    
    s = sun(city.observer, date=datetime.date(year, month, day))
    seconds = (s['sunset'] - s['sunrise']).seconds    
    minute = np.round(seconds/60,0)
    
    return minute
df_gen['LIGHT_DURING_DAY'] = df_gen['DATE_TIME'].apply(lambda x: light_during_day(x))
print('Maximum time: {} minutes'.format(df_gen['LIGHT_DURING_DAY'].max()))
print('Minimum time: {} minutes'.format(df_gen['LIGHT_DURING_DAY'].min()))
df_sensor.info()
profile_sensor = ProfileReport(df_sensor)
profile_sensor.to_widgets()
profile_sensor.to_notebook_iframe()
# df_sensor['DATE_TIME'] = pd.to_datetime(df_sensor['DATE_TIME'])

from datetime import datetime

df_sensor['DATE_TIME'] = df_sensor['DATE_TIME'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
df_sensor['DATE_TIME'].max()
df_sensor['DATE_TIME'].min()
df_sensor.head()
# Remove PLANT_ID due to it has only 1 distinct value

df_sensor = df_sensor.drop('PLANT_ID', axis = 1)
df_sensor.head()
# Remove SOURCE_KEY due to it has only 1 distinct value

df_sensor = df_sensor.drop('SOURCE_KEY',axis = 1)
df_sensor.head()
plt.figure(figsize=(20, 8))

sns.lineplot(data=df_sensor, x="DATE_TIME", y="AMBIENT_TEMPERATURE")
plt.figure(figsize=(20, 8))

sns.lineplot(data=df_sensor, x="DATE_TIME", y="MODULE_TEMPERATURE")
plt.figure(figsize=(20, 8))

sns.lineplot(data=df_sensor, x="DATE_TIME", y="IRRADIATION")
df_join = pd.merge(df_gen, df_sensor, left_on='DATE_TIME', right_on='DATE_TIME', how='left')
df_join.head()
df_join.info()
df_join[(df_join['AMBIENT_TEMPERATURE'].isna()) & df_join['AC_POWER'] != 0]
plt.figure(figsize=(20, 8))

sns.lineplot(data=df_join, x="DATE_TIME", y='AMBIENT_TEMPERATURE', hue='SOURCE_KEY')
df_join.iloc[38543:38549]
df_join.fillna(method='ffill', inplace=True)
df_join.iloc[38543:38549]
df_join.info()
# df_join['AMBIENT_TEMPERATURE_d-1'] = df_join.sort_values(by=['SOURCE_KEY','TIME','DATE'], ascending=False).shift(-1)['AMBIENT_TEMPERATURE']
# df_join['AMBIENT_TEMPERATURE_d-2'] = df_join.sort_values(by=['SOURCE_KEY','TIME','DATE'], ascending=False).shift(-2)['AMBIENT_TEMPERATURE']
# df_join['AMBIENT_TEMPERATURE_d-3'] = df_join.sort_values(by=['SOURCE_KEY','TIME','DATE'], ascending=False).shift(-3)['AMBIENT_TEMPERATURE']
# df_join.drop('AMBIENT_TEMPERATURE_d-3', axis = 1, inplace = True)

def create_past_col(df, col_name, day_back):
    for day in range(-day_back,0):
        df[col_name + '_d' + str(day)] = df.sort_values(by=['SOURCE_KEY','TIME','DATE'], ascending=False).shift(day)[col_name]
        print('Finish column: {}{}'.format(col_name,day))
df_join_3day = df_join.copy()
create_past_col(df_join_3day, 'AMBIENT_TEMPERATURE', 3)
create_past_col(df_join_3day, 'AC_POWER', 3)
create_past_col(df_join_3day, 'TOTAL_YIELD', 3)
create_past_col(df_join_3day, 'LIGHT_DURING_DAY', 3)
create_past_col(df_join_3day, 'MODULE_TEMPERATURE', 3)
create_past_col(df_join_3day, 'IRRADIATION', 3)
df_join_3day.head()
df_join_3day.sort_values(by=['SOURCE_KEY','TIME','DATE'], ascending=False).tail()
df_join_3day.info()
df_join_3day[df_join_3day['TOTAL_YIELD_d-1'].isna()]
# df_join_3day = df_join_3day.dropna()

df_join_3day = df_join_3day.fillna(df_join_3day.mean())
col_to_drop = ['AMBIENT_TEMPERATURE','AC_POWER','LIGHT_DURING_DAY','MODULE_TEMPERATURE','IRRADIATION','DATE_TIME',\
              'SOURCE_KEY','DATE','TIME']


df_join_3day = df_join_3day.drop(col_to_drop, axis = 1)
df_join_3day.head()
df_join_7day = df_join.copy()
create_past_col(df_join_7day, 'AMBIENT_TEMPERATURE', 7)
create_past_col(df_join_7day, 'AC_POWER', 7)
create_past_col(df_join_7day, 'TOTAL_YIELD', 7)
create_past_col(df_join_7day, 'LIGHT_DURING_DAY', 7)
create_past_col(df_join_7day, 'MODULE_TEMPERATURE', 7)
create_past_col(df_join_7day, 'IRRADIATION', 7)
df_join_7day = df_join_7day.drop(col_to_drop, axis = 1)
df_join_7day.head()
df_join_7day.info()
# df_join_7day = df_join_7day.dropna()

df_join_7day = df_join_7day.fillna(df_join_7day.mean())
def modeling(X_train,Y_train,X_test,Y_test):
    mse_list = []
    rmse_list = []
    mse_dict = {}
    rmse_dict = {}
    
    ###############################
    # Linear Regression
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, Y_train)    
    y_pred_lr = linear_regressor.predict(X_test)
    
    mse_lr = mean_squared_error(Y_test, y_pred_lr)
    mse_list.append(mse_lr)
    rmse_lr = sqrt(mse_lr)
    rmse_list.append(rmse_lr)
    mse_dict['Linear_Regression'] = mse_lr
    rmse_dict['Linear_Regression'] = rmse_lr
    
    ###############################
    # Polynomial Regression
    poly_reg = PolynomialFeatures(degree = 3)
    X_poly = poly_reg.fit_transform(X_train)

    poly_regressor = LinearRegression()
    poly_regressor.fit(X_poly, Y_train)

    X_poly_test = poly_reg.fit_transform(X_test)
    y_pred_pr = poly_regressor.predict(X_poly_test)
    
    mse_pr = mean_squared_error(Y_test, y_pred_pr)    
    mse_list.append(mse_pr)
    rmse_pr = sqrt(mse_pr)
    rmse_list.append(rmse_pr)    
    mse_dict['Polynomial_Regression'] = mse_pr
    rmse_dict['Polynomial_Regression'] = rmse_pr
    
    ###############################
    # Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
    rf_regressor.fit(X_train, Y_train)
    y_pred_rfr = rf_regressor.predict(X_test)
    
    mse_rfr = mean_squared_error(Y_test, y_pred_rfr)
    mse_list.append(mse_rfr)
    rmse_rfr = sqrt(mse_rfr)
    rmse_list.append(rmse_rfr)    
    mse_dict['Random_Forest_Regressor'] = mse_rfr
    rmse_dict['Random_Forest_Regressor'] = rmse_rfr
    
    ###############################
    # LGBM
#     hyper_params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'metric': ['rmse'],
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.7,
#     'bagging_freq': 10,
#     'verbose': -1,
#     "max_depth": 8,
#     "num_leaves": 128,  
#     "max_bin": 256,
#     "num_iterations": 50,
#     "n_estimators": 1000
#     }

#     gbm = lgb.LGBMRegressor(verbose_eval=False, **hyper_params)

#     gbm.fit(X_train, Y_train,
#             eval_set=[(X_test, Y_test)],
#             eval_metric='rmse',
#             early_stopping_rounds=50)

#     Y_pred_lgbm = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    
#     mse_lgbm = mean_squared_error(Y_test, Y_pred_lgbm)
#     mse_list.append(mse_lgbm)
#     rmse_lgbm = sqrt(mse_lgbm)
#     rmse_list.append(rmse_lgbm)
#     mse_dict['LightGBM'] = mse_lgbm
#     rmse_dict['LightGBM'] = rmse_lgbm
        
    return mse_list, rmse_list, mse_dict, rmse_dict
df_join_3day.iloc[0::10].head()
# X_train = df_join_3day.drop(df_join_3day.index[1::10]).drop('TOTAL_YIELD', axis = 1)
# Y_train = df_join_3day.drop(df_join_3day.index[1::10], errors='ignore')['TOTAL_YIELD']

# X_test = df_join_3day.iloc[1::10].drop('TOTAL_YIELD', axis = 1)
# Y_test = df_join_3day.iloc[1::10]['TOTAL_YIELD']
for k in range(10):

    X_train = df_join_3day.drop(df_join_3day.index[k::10]).drop('TOTAL_YIELD', axis = 1)
    Y_train = df_join_3day.drop(df_join_3day.index[k::10], errors='ignore')['TOTAL_YIELD']
    
    X_test = df_join_3day.iloc[k::10].drop('TOTAL_YIELD', axis = 1)
    Y_test = df_join_3day.iloc[k::10]['TOTAL_YIELD']
    
    mse_list, rmse_list, mse_dict, rmse_dict = modeling(X_train,Y_train,X_test,Y_test)
    
    print('############################\n')
    print('fold: {}\n'.format(k))
    
    print('MSE:')
    print(mse_dict)
    
    print('RMSE:')
    print(rmse_dict)
    
    print('\n############################\n')
for k in range(10):


    X_train = df_join_3day.drop(df_join_3day.index[k::10]).drop(['TOTAL_YIELD','TOTAL_YIELD_d-1','TOTAL_YIELD_d-2','TOTAL_YIELD_d-3'], axis = 1)
    Y_train = df_join_3day.drop(df_join_3day.index[k::10], errors='ignore')['TOTAL_YIELD']
    
    X_test = df_join_3day.iloc[k::10].drop(['TOTAL_YIELD','TOTAL_YIELD_d-1','TOTAL_YIELD_d-2','TOTAL_YIELD_d-3'], axis = 1)
    Y_test = df_join_3day.iloc[k::10]['TOTAL_YIELD']
    
    mse_list, rmse_list, mse_dict, rmse_dict = modeling(X_train,Y_train,X_test,Y_test)
    
    print('############################\n')
    print('fold: {}'.format(k))
    
    print('MSE:')
    print(mse_dict)
    
    print('RMSE:')
    print(rmse_dict)
    
    print('\n############################\n')
def modeling_7days(X_train,Y_train,X_test,Y_test):
    mse_list = []
    rmse_list = []
    mse_dict = {}
    rmse_dict = {}
    
    ###############################
    # Linear Regression
#     linear_regressor = LinearRegression()
#     linear_regressor.fit(X_train, Y_train)    
#     y_pred_lr = linear_regressor.predict(X_test)
    
#     mse_lr = mean_squared_error(Y_test, y_pred_lr)
#     mse_list.append(mse_lr)
#     rmse_lr = sqrt(mse_lr)
#     rmse_list.append(rmse_lr)
#     mse_dict['Linear_Regression'] = mse_lr
#     rmse_dict['Linear_Regression'] = rmse_lr
    
    ###############################
    # Polynomial Regression
#     poly_reg = PolynomialFeatures(degree = 3)
#     X_poly = poly_reg.fit_transform(X_train)

#     poly_regressor = LinearRegression()
#     poly_regressor.fit(X_poly, Y_train)

#     X_poly_test = poly_reg.fit_transform(X_test)
#     y_pred_pr = poly_regressor.predict(X_poly_test)
    
#     mse_pr = mean_squared_error(Y_test, y_pred_pr)    
#     mse_list.append(mse_pr)
#     rmse_pr = sqrt(mse_pr)
#     rmse_list.append(rmse_pr)    
#     mse_dict['Polynomial_Regression'] = mse_pr
#     rmse_dict['Polynomial_Regression'] = rmse_pr
    
    ###############################
    # Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
    rf_regressor.fit(X_train, Y_train)
    y_pred_rfr = rf_regressor.predict(X_test)
    
    mse_rfr = mean_squared_error(Y_test, y_pred_rfr)
    mse_list.append(mse_rfr)
    rmse_rfr = sqrt(mse_rfr)
    rmse_list.append(rmse_rfr)    
    mse_dict['Random_Forest_Regressor'] = mse_rfr
    rmse_dict['Random_Forest_Regressor'] = rmse_rfr
    
    ###############################
    # LGBM
#     hyper_params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'metric': ['rmse'],
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.7,
#     'bagging_freq': 10,
#     'verbose': -1,
#     "max_depth": 8,
#     "num_leaves": 128,  
#     "max_bin": 256,
#     "num_iterations": 50,
#     "n_estimators": 1000
#     }

#     gbm = lgb.LGBMRegressor(verbose_eval=False, **hyper_params)

#     gbm.fit(X_train, Y_train,
#             eval_set=[(X_test, Y_test)],
#             eval_metric='rmse',
#             early_stopping_rounds=50)

#     Y_pred_lgbm = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    
#     mse_lgbm = mean_squared_error(Y_test, Y_pred_lgbm)
#     mse_list.append(mse_lgbm)
#     rmse_lgbm = sqrt(mse_lgbm)
#     rmse_list.append(rmse_lgbm)
#     mse_dict['LightGBM'] = mse_lgbm
#     rmse_dict['LightGBM'] = rmse_lgbm
        
    return mse_list, rmse_list, mse_dict, rmse_dict
for k in range(10):

    X_train = df_join_7day.drop(df_join_7day.index[k::10]).drop('TOTAL_YIELD', axis = 1)
    Y_train = df_join_7day.drop(df_join_7day.index[k::10], errors='ignore')['TOTAL_YIELD']
    
    X_test = df_join_7day.iloc[k::10].drop('TOTAL_YIELD', axis = 1)
    Y_test = df_join_7day.iloc[k::10]['TOTAL_YIELD']
    
    mse_list, rmse_list, mse_dict, rmse_dict = modeling_7days(X_train,Y_train,X_test,Y_test)
    
    print('############################\n')
    print('fold: {}'.format(k))   
    
    print('MSE:')    
    print(mse_dict)
    
    print('RMSE:') 
    print(rmse_dict)
    
    print('\n############################\n')
for k in range(10):

    X_train = df_join_7day.drop(df_join_7day.index[k::10]).drop(['TOTAL_YIELD','TOTAL_YIELD_d-1','TOTAL_YIELD_d-2','TOTAL_YIELD_d-3',\
                                                                 'TOTAL_YIELD_d-4','TOTAL_YIELD_d-5','TOTAL_YIELD_d-6','TOTAL_YIELD_d-7'], axis = 1)
    Y_train = df_join_7day.drop(df_join_7day.index[k::10], errors='ignore')['TOTAL_YIELD']
    
    X_test = df_join_7day.iloc[k::10].drop(['TOTAL_YIELD','TOTAL_YIELD_d-1','TOTAL_YIELD_d-2','TOTAL_YIELD_d-3',\
                                           'TOTAL_YIELD_d-4','TOTAL_YIELD_d-5','TOTAL_YIELD_d-6','TOTAL_YIELD_d-7'], axis = 1)
    Y_test = df_join_7day.iloc[k::10]['TOTAL_YIELD']
    
    mse_list, rmse_list, mse_dict, rmse_dict = modeling_7days(X_train,Y_train,X_test,Y_test)
    
    print('############################\n')
    print('fold: {}'.format(k))
    
    print('MSE:')    
    print(mse_dict)
    
    print('RMSE:')  
    print(rmse_dict)
    
    print('\n############################\n')