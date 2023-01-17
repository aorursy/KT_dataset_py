# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier

from google.cloud import bigquery

from scipy.spatial.distance import cdist

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# загружаем данные тренировочного набора



train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
# Импортируем данные признаков из https://www.kaggle.com/davidbnn92/weather-data?scriptVersionId=30695168

client = bigquery.Client()

dataset_ref = client.dataset("noaa_gsod", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))



table_ref = dataset_ref.table("stations")

table = client.get_table(table_ref)

stations_df = client.list_rows(table).to_dataframe()



table_ref = dataset_ref.table("gsod2020")

table = client.get_table(table_ref)

twenty_twenty_df = client.list_rows(table).to_dataframe()



stations_df['STN'] = stations_df['usaf'] + '-' + stations_df['wban']

twenty_twenty_df['STN'] = twenty_twenty_df['stn'] + '-' + twenty_twenty_df['wban']



cols_1 = ['STN', 'mo', 'da', 'temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']

cols_2 = ['STN', 'country', 'state', 'call', 'lat', 'lon', 'elev']

weather_df = twenty_twenty_df[cols_1].join(stations_df[cols_2].set_index('STN'), on='STN')



weather_df.tail(10)
weather_df['day_from_jan_first'] = (weather_df['da'].apply(int)

                                   + 31*(weather_df['mo']=='02') 

                                   + 60*(weather_df['mo']=='03')

                                   + 91*(weather_df['mo']=='04')  

                                   )



mo = train['Date'].apply(lambda x: x[5:7])

da = train['Date'].apply(lambda x: x[8:10])

train['day_from_jan_first'] = (da.apply(int)

                               + 31*(mo=='02') 

                               + 60*(mo=='03')

                               + 91*(mo=='04')  

                              )



C = []

for j in train.index:

    df = train.iloc[j:(j+1)]

    mat = cdist(df[['Lat','Long', 'day_from_jan_first']],

                weather_df[['lat','lon', 'day_from_jan_first']], 

                metric='euclidean')

    new_df = pd.DataFrame(mat, index=df.Id, columns=weather_df.index)

    arr = new_df.values

    new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)

    L = [i[i.astype(bool)].tolist()[0] for i in new_close]

    C.append(L[0])

    

train['closest_station'] = C



train = train.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)

train.sort_values(by=['Id'], inplace=True)

train.head()
# загружаем тестовый набор данных

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")
weather_df['day_from_jan_first'] = (weather_df['da'].apply(int)

                                   + 31*(weather_df['mo']=='02') 

                                   + 60*(weather_df['mo']=='03')

                                   + 91*(weather_df['mo']=='04')  

                                   )



mo = test['Date'].apply(lambda x: x[5:7])

da = test['Date'].apply(lambda x: x[8:10])

test['day_from_jan_first'] = (da.apply(int)

                               + 31*(mo=='02') 

                               + 60*(mo=='03')

                               + 91*(mo=='04')  

                              )



C = []

for j in test.index:

    df = test.iloc[j:(j+1)]

    mat = cdist(df[['Lat','Long', 'day_from_jan_first']],

                weather_df[['lat','lon', 'day_from_jan_first']], 

                metric='euclidean')

    new_df = pd.DataFrame(mat, index=df.ForecastId, columns=weather_df.index)

    arr = new_df.values

    new_close = np.where(arr == np.nanmin(arr, axis=1)[:,None],new_df.columns,False)

    L = [i[i.astype(bool)].tolist()[0] for i in new_close]

    C.append(L[0])

    

test['closest_station'] = C



test = test.set_index('closest_station').join(weather_df[['temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog']], ).reset_index().drop(['index'], axis=1)

test.sort_values(by=['ForecastId'], inplace=True)

test.head()
# изменяем тип столбцов wdsp

train["wdsp"] = pd.to_numeric(train["wdsp"])

test["wdsp"] = pd.to_numeric(test["wdsp"])

# изменяем тип столбцов fog

train["fog"] = pd.to_numeric(train["fog"])

test["fog"] = pd.to_numeric(test["fog"])
# удалим из признаков столбцы для прогнозирования

X_train = train.drop(["Fatalities", "ConfirmedCases", "Id"], axis=1)

X_test = test.drop(["ForecastId"], axis=1)
X_train.info()
# приведем значение даты в тип datetime

X_train['Date']= pd.to_datetime(X_train['Date']) 

X_test['Date']= pd.to_datetime(X_test['Date']) 
# установим индекс на дату

X_train = X_train.set_index(['Date'])

X_test = X_test.set_index(['Date'])
def create_time_features(df):

    """Создает элементы временного ряда из индекса даты и времени."""

    df['date'] = df.index

    df['hour'] = df['date'].dt.hour

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    X = df[['hour','dayofweek','quarter','month','year',

           'dayofyear','dayofmonth','weekofyear']]

    return X
# Создает элементы временного ряда

create_time_features(X_train)

create_time_features(X_test)
X_train.head()
# удалим столбецц с датой, так как мы уже создали данные временного ряда

X_train.drop("date", axis=1, inplace=True)

X_test.drop("date", axis=1, inplace=True)
# One-Hot-encoding для городов

X_train = pd.concat([X_train,pd.get_dummies(X_train['Province/State'], prefix='ps')],axis=1)

X_train.drop(['Province/State'],axis=1, inplace=True)

X_test = pd.concat([X_test,pd.get_dummies(X_test['Province/State'], prefix='ps')],axis=1)

X_test.drop(['Province/State'],axis=1, inplace=True)

# One-Hot-encoding для стран

X_train = pd.concat([X_train,pd.get_dummies(X_train['Country/Region'], prefix='cr')],axis=1)

X_train.drop(['Country/Region'],axis=1, inplace=True)

X_test = pd.concat([X_test,pd.get_dummies(X_test['Country/Region'], prefix='cr')],axis=1)

X_test.drop(['Country/Region'],axis=1, inplace=True)
Y1= train["ConfirmedCases"]
Y2 = train["Fatalities"]
# определяем количество заболевших

model = RandomForestClassifier(bootstrap=True,max_depth=None, max_features='auto', max_leaf_nodes=None, 

                      n_estimators=150, random_state=None, n_jobs=1, verbose=0)

model.fit(X_train,Y1)

pred1 = model.predict(X_test)

pred1 = pd.DataFrame(pred1)

pred1.columns = ["ConfirmedCases_prediction"]
pred1[50:100]
# определяем количество умерших

model = RandomForestClassifier(bootstrap=True,max_depth=None, max_features='auto', max_leaf_nodes=None, 

                      n_estimators=150, random_state=None, n_jobs=1, verbose=0)

model.fit(X_train,Y2)

pred2 = model.predict(X_test)

pred2 = pd.DataFrame(pred2)

pred2.columns = ["Death_prediction"]
pred2.head()
data_submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/submission.csv")

data_submission.columns

sub_new = data_submission[["ForecastId"]]
concat = pd.concat([pred1,pred2,sub_new],axis=1)

concat.head()

concat.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

concat = concat[['ForecastId','ConfirmedCases', 'Fatalities']]
concat["ConfirmedCases"] = concat["ConfirmedCases"].astype(int)

concat["Fatalities"] = concat["Fatalities"].astype(int)
concat.head()
concat.to_csv("submission.csv",index=False)