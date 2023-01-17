# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from collections import Counter

from datetime import datetime

from google.cloud import bigquery

import matplotlib.pyplot as plt

import os
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
train.rename(columns = {'Province_State':'Province','Country_Region':'Country'},inplace = True)

test.rename(columns = {'Province_State':'Province','Country_Region':'Country'},inplace = True)
for i in range(len(train)):

    if train.Province[i] is np.NaN:

        train.Province[i] = train.Country[i]

for i in range(len(test)):

    if test.Province[i] is np.NaN:

        test.Province[i] = test.Country[i]
def get_days(x):

    x = datetime.strptime(x,'%Y-%m-%d')

    first = datetime.strptime('2020-01-01','%Y-%m-%d')

    diff = (x - first).days

    return diff
train['Days_After_1stJan'] = train.Date.apply(lambda x :get_days(x))

test['Days_After_1stJan'] = test.Date.apply(lambda x :get_days(x))
train.head()
test.head()
def get_dt(x):

    x = datetime.strptime(x,'%Y-%m-%d')

    return x

def get_dayofweek(x):

    return x.dayofweek

def get_month(x):

    return x.month

def get_day(x):

    return x.day
train['Date'] = train.Date.apply(lambda x:get_dt(x))

test['Date'] = test.Date.apply(lambda x:get_dt(x))

train['Dayofweek'] = train.Date.apply(lambda x:get_dayofweek(x))

test['Dayofweek'] = test.Date.apply(lambda x:get_dayofweek(x))

train['Month'] = train.Date.apply(lambda x:get_month(x))

test['Month'] = test.Date.apply(lambda x:get_month(x))

train['Day'] = train.Date.apply(lambda x:get_day(x))

test['Day'] = test.Date.apply(lambda x:get_day(x))
country_info = pd.read_csv('/kaggle/input/population/population_by_country_2020.csv')
population = pd.DataFrame(country_info.iloc[:,[0,1,4,5,6,8,9]])

population.columns = ['Country','Population','Density','Land_Area','Migrants','MedAge','UrbanPopRate']
for i in range(len(population)):

    if np.isnan(population.Migrants[i]):

        population.Migrants[i] = np.nanmedian(population.Migrants)

    if population.MedAge[i] == 'N.A.':

        population.MedAge[i] = 19

    if population.UrbanPopRate[i] == 'N.A.':

        population.UrbanPopRate[i] = '57%'
kosovo = pd.DataFrame([['Kosovo'],[2000700],[168],[10887],[0],[19],['57%']])

kosovo = kosovo.T

kosovo.columns = population.columns





population = population.append(kosovo)
westbank = pd.DataFrame([['West Bank and Gaza'],[2697687],[485],[5559],[0],[19],['57%']])

westbank = westbank.T

westbank.columns = population.columns



population = population.append(westbank)
DP = pd.DataFrame(['Diamond Princess',2666,191522,0.01392,2666,19,'100%'])

DP = DP.T

DP.columns = population.columns



population = population.append(DP)
Shangdan = pd.DataFrame(['MS Zaandam',1432,189618,0.007552,1432,19,'100%'])

Shangdan = Shangdan.T

Shangdan.columns = population.columns



population = population.append(Shangdan)
Congo = population[population.Country == 'Congo']

Congo['Country'] = 'Congo (Brazzaville)'

new1 = Congo.copy()

Congo['Country'] = 'Congo (Kinshasa)'

new2 = Congo.copy()



population = population.append(new1)

population = population.append(new2)
population.Country[population.Country == 'United States'] = 'US'

population.Country[population.Country == 'Taiwan'] = 'Taiwan*'

population.Country[population.Country == 'South Korea'] = 'Korea, South'

population.Country[population.Country == 'Côte d\'Ivoire'] = 'Cote d\'Ivoire'

population.Country[population.Country == 'Czech Republic (Czechia)'] = 'Czechia'

population.Country[population.Country == 'Myanmar'] = 'Burma'

population.Country[population.Country == 'St. Vincent & Grenadines'] = 'Saint Vincent and the Grenadines'

population.Country[population.Country == 'Saint Kitts & Nevis']  = 'Saint Kitts and Nevis'

population.Country[population.Country == 'Sao Tome & Principe']  = 'Sao Tome and Principe'
train = pd.merge(train,population,left_on = 'Country',right_on='Country',how='left')

test = pd.merge(test,population,left_on = 'Country',right_on='Country',how = 'left')
train.head()
test.head()
temperature = pd.read_csv('/kaggle/input/weather-data-for-covid19-data-analysis/training_data_with_weather_info_week_4.csv')

temperature.rename(columns = {'Province_State':'Province','Country_Region':'Country'},inplace = True)
# fill NAN of Province with Country name

for i in range(len(temperature)):

    if temperature.Province[i] is np.NaN:

        temperature.Province[i] = temperature.Country[i]
#delete useless features 

del temperature['ConfirmedCases']

del temperature['Fatalities']

del temperature['country+province']

del temperature['day_from_jan_first']
#transform date to datetime type

temperature.Date = temperature.Date.apply(lambda x:get_dt(x))
day1 = datetime.strptime('2020-04-09','%Y-%m-%d')

day2 = datetime.strptime('2020-04-10','%Y-%m-%d')

day3 = datetime.strptime('2020-04-11','%Y-%m-%d')

day4 = datetime.strptime('2020-04-12','%Y-%m-%d')
index_delete = []

for i in range(len(train)):

    if (train.Date[i] == day1) or (train.Date[i] == day2) or (train.Date[i] == day3) or (train.Date[i] == day4):

        index_delete.append(i)

        

train = (train.drop(index = index_delete)).reset_index(drop = True)
train = pd.merge(train,temperature,on=['Country','Province','Date'],how='left')
#fill NAN

train['Lat'][train['Lat'].isnull()] = np.nanmedian(train['Lat'])

train['Long'][train['Long'].isnull()] = np.nanmedian(train['Long'])

train['temp'][train['temp'].isnull()] = np.nanmedian(train['temp'])

train['min'][train['min'].isnull()] = np.nanmedian(train['min'])

train['max'][train['max'].isnull()] = np.nanmedian(train['max'])

train['slp'][train['slp'].isnull()] = np.nanmedian(train['slp'])

train['dewp'][train['dewp'].isnull()] = np.nanmedian(train['dewp'])

train['rh'][train['rh'].isnull()] = np.nanmedian(train['rh'])

train['ah'][train['ah'].isnull()] = np.nanmedian(train['ah'])

train['stp'][train['stp'].isnull()] = np.nanmedian(train['stp'])

train['wdsp'][train['wdsp'].isnull()] = np.nanmedian(train['wdsp'])

train['prcp'][train['prcp'].isnull()] = np.nanmedian(train['prcp'])

train['fog'][train['fog'].isnull()] = np.nanmedian(train['fog'])
!pip install pmdarima
import pmdarima
#the outlier 'inf' will make the auto_arima come to an error,so it's replaced by 0

train.replace(np.inf,0,inplace=True)
#Using arima to predict the weather information for future

date_pred_df = pd.DataFrame(sorted(list(set(test.Date))),columns=['Date'])

date_pred_df = date_pred_df[7:]

date_pred_df.reset_index(inplace = True)

del date_pred_df['index']



nperiods = (datetime.strptime('2020-05-14','%Y-%m-%d')-datetime.strptime('2020-04-08','%Y-%m-%d')).days



weather_feature = ['temp', 'min', 'max', 'stp', 'slp', 'dewp', 'rh', 'ah','wdsp', 'prcp', 'fog']

weather_pred = pd.DataFrame(columns=['Date','temp', 'min', 'max', 'stp', 'slp', 'dewp', 'rh', 'ah','wdsp', 'prcp', 'fog'])

for prov in list(set(train.Province)):

    df = train[train.Province == prov]

    province_pred = date_pred_df.copy()

    for feature in weather_feature:

        ts = df[feature]

        model = pmdarima.auto_arima(ts)

        pred = model.predict(n_periods = nperiods)

        province_pred[feature] = pred

    province_pred['Province'] = prov

    weather_pred = pd.concat([weather_pred,province_pred],axis = 0)
test
for i in range(len(test)):

    if test.Date[i]<datetime.strptime('2020-04-09','%Y-%m-%d'):

        test.drop(i,inplace=True)
#get longitude and latitude dataframe and merge it into test df

df_longlat = pd.DataFrame(columns = ['Province','Lat','Long'])

for i in range(len(train)):

    if train.Province[i] not in list(df_longlat.Province):

        df_longlat = df_longlat.append(train.iloc[i][['Province','Lat','Long']])

        

test = pd.merge(test,df_longlat,on = 'Province',how = 'left')
#adding weather feature to test data

test = pd.merge(test,weather_pred,on = ['Province','Date'],how = 'left')
train.head()
test.head()
API_beds = pd.read_csv('/kaggle/input/newest-bed-api-for-each-country/Newest_avg_bed_API.csv')
#merge

train = pd.merge(train,API_beds,left_on='Country',right_on='Country',how='left')

test = pd.merge(test,API_beds,left_on='Country',right_on='Country',how='left')
#fill NAN

train.API_beds[train.API_beds.isnull()] = np.nanmedian(train.API_beds)

test.API_beds[test.API_beds.isnull()] = np.nanmedian(test.API_beds)
train.head()
test.head()
X = train.copy()

X_test = test.copy()



Province_set = set(X.Province)

Country_set = set(X.Country)



X = pd.concat([X,pd.get_dummies(X.Country)],axis=1)

X_test = pd.concat([X_test,pd.get_dummies(X_test.Country)],axis=1)

X = pd.concat([X,pd.get_dummies(X.Province)[Province_set - Country_set]],axis=1)

X_test = pd.concat([X_test,pd.get_dummies(X_test.Province)[Province_set - Country_set]],axis=1)



y_confirm = X.ConfirmedCases

y_fata = X.Fatalities

del X['ConfirmedCases']

del X['Fatalities']

del X['Id_x']

del X['Date']

del X['Id_y']

del X['Province']

del X['Country']
ForecastId = X_test.ForecastId
def get_percent(x):

    x = str(x)

    x = x.strip('%')

    x = float(x)/100

    return x
X['UrbanPopRate'] = X.UrbanPopRate.apply(lambda x:get_percent(x))

X_test['UrbanPopRate'] = X_test.UrbanPopRate.apply(lambda x:get_percent(x))
del X_test['ForecastId']

del X_test['Date']

del X_test['Province']

del X_test['Country']
X = pd.DataFrame(X,dtype=float)

X_test = pd.DataFrame(X_test,dtype=float)
order = X.columns

X_test = X_test[order]
X.head()
X_test.head()
import xgboost as xgb
reg_confirm = xgb.XGBRegressor()

reg_confirm.fit(X,y_confirm)
pred_confirm = reg_confirm.predict(X_test)
reg_fata = xgb.XGBRegressor()

reg_fata.fit(X,y_fata)
pred_fata = reg_fata.predict(X_test)
submit = pd.DataFrame(ForecastId)

submit['ConfirmedCases']=pred_confirm

submit['Fatalities']=pred_fata
temp_submit_df = submit.copy()

temp_submit_df['Province'] = test.Province

temp_submit_df['Date'] = test['Date']

del temp_submit_df['ForecastId']
original_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

new_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
original_train.Date = original_train.Date.apply(lambda x:get_dt(x))

original_train.rename(columns = {'Province_State':'Province','Country_Region':'Country'},inplace = True)

for i in range(len(original_train)):

    if original_train.Province[i] is np.NaN:

        original_train.Province[i] = original_train.Country[i]

        

for i in range(len(original_train)):

    if original_train.Date[i]<datetime.strptime('2020-04-02','%Y-%m-%d') or original_train.Date[i]>datetime.strptime('2020-04-08','%Y-%m-%d'):

        original_train.drop(i,inplace=True)

        

del original_train['Id']

del original_train['Country']
final = pd.concat([temp_submit_df,original_train],axis = 0)

final = final.sort_values(by=['Province','Date'])

final_submit = pd.DataFrame(new_test.ForecastId,columns=['ForecastId'])

final_submit['Confirmed'] = final.ConfirmedCases.values

final_submit['Fatalities'] = final.Fatalities.values
final_submit
final_submit.to_csv('/kaggle/working/submission.csv',index=False)