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

pd.options.display.max_rows=1000
#train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

#train = pd.read_csv('/kaggle/input/train-week2-doctored/trainDoctored.csv')

train = pd.read_csv('/kaggle/input/train-proc/train_proc.csv')

#test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

test = pd.read_csv('/kaggle/input/test-proc/test_proc.csv')

subs = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
cow = pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv')
cow.rename(columns={'Country':'Country_Region'},inplace=True)
cow['Pop. Density (per sq. mi.)'] = cow['Pop. Density (per sq. mi.)'].str.replace(',', '.').astype(float)

cow['Coastline (coast/area ratio)'] = cow['Coastline (coast/area ratio)'].str.replace(',', '.').astype(float)

cow['Net migration'] = cow['Net migration'].str.replace(',', '.').astype(float)

cow['Infant mortality (per 1000 births)'] = cow['Infant mortality (per 1000 births)'].str.replace(',', '.').astype(float)

cow['Literacy (%)'] = cow['Literacy (%)'].str.replace(',', '.').astype(float)

cow['Phones (per 1000)'] = cow['Phones (per 1000)'].str.replace(',', '.').astype(float)

cow['Arable (%)'] = cow['Arable (%)'].str.replace(',', '.').astype(float)

cow['Crops (%)'] = cow['Crops (%)'].str.replace(',', '.').astype(float)

cow['Other (%)'] = cow['Other (%)'].str.replace(',', '.').astype(float)

cow['Climate'] = cow['Climate'].str.replace(',', '.').astype(float)

cow['Birthrate'] = cow['Birthrate'].str.replace(',', '.').astype(float)

cow['Deathrate'] = cow['Deathrate'].str.replace(',', '.').astype(float)

cow['Agriculture'] = cow['Agriculture'].str.replace(',', '.').astype(float)

cow['Industry'] = cow['Industry'].str.replace(',', '.').astype(float)

cow['Service'] = cow['Service'].str.replace(',', '.').astype(float)
# !pip install geopy
# train['Province_State'].fillna('',inplace=True)

# train_groupped = train.groupby(['Province_State','Country_Region']).max().reset_index()

# #train_groupped['Probab'] = train_groupped['Fatalities']/train_groupped['ConfirmedCases']

# train_groupped = train_groupped.sort_values(by='Country_Region')

# train_groupped
# train_groupped['Lat']=None

# train_groupped['Long']=None

# from geopy.geocoders import Nominatim

# geolocator = Nominatim(user_agent="specify_your_app_name_here")

# for i in range(len(train_groupped)):

#     if train_groupped['Province_State'].iloc[i]=='':

#         loc = train_groupped['Country_Region'].iloc[i]

#     else:

#         loc = train_groupped['Province_State'].iloc[i] + ', ' + train_groupped['Country_Region'].iloc[i]

#     location = geolocator.geocode(loc)

#     try:

#         train_groupped['Lat'].iloc[i] = location.latitude

#         train_groupped['Long'].iloc[i] =  location.longitude

#     except:

#         loc = train_groupped['Country_Region'].iloc[i]

#         location = geolocator.geocode(loc)

#         train_groupped['Lat'].iloc[i] = location.latitude

#         train_groupped['Long'].iloc[i] =  location.longitude

        

# bcg_countries = ['Armenia','Azerbaijan','Belarus','Estonia','Georgia','Kazakhstan','Kyrgyzstan','Latvia','Lithuania',

#                  'Moldova','Russia','Tajikistan','Turkmenistan','Ukraine','Uzbekistan',

#                 'United Kingdom','India','Brazil','Germany','Bulgaria','Hungary', 'Poland','Romania','Slovakia',

#                  'Malta','France','Norway','Greece','Singapore','Malaysia', 'Korea, South', 'Taiwan','Japan','Thailand','Sri Lanka','South Africa',

#                 'Philippines','Mongolia','Hong Kong','Pakistan','Ecuador','Argentina','Bolivia','Columbia','Chile','Paraguay','Peru','Uruguay','Venezuela']



# train_groupped['BCG'] = [i in  bcg_countries for i in train_groupped['Country_Region']]

# train = pd.merge(train,train_groupped[['Province_State','Country_Region','Lat','Long','BCG']])

# train = train.sort_values(by='Date').reset_index(drop=True)

# train.to_csv('train_proc.csv')
cow[cow['Country_Region']=='United States']['Country_Region']='US '

cow.fillna(0,inplace=True)
train['Province_State'].fillna(train['Country_Region'],inplace=True)

test['Province_State'].fillna(test['Country_Region'],inplace=True)
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])

train['Date'] = train['Date'].astype('int64')

test['Date'] = test['Date'].astype('int64')
train['Country_Region'] = train['Country_Region'].apply(lambda x: str(x)+' ')

test['Country_Region'] = test['Country_Region'].apply(lambda x: str(x)+' ')
# test['Province_State'].fillna('',inplace=True)

# test_groupped = test.groupby(['Province_State','Country_Region']).max().reset_index()



# test_groupped['Lat']=None

# test_groupped['Long']=None

# from geopy.geocoders import Nominatim

# geolocator = Nominatim(user_agent="specify_your_app_name_here")

# for i in range(len(test_groupped)):

#     if test_groupped['Province_State'].iloc[i]=='':

#         loc = test_groupped['Country_Region'].iloc[i]

#     else:

#         loc = test_groupped['Province_State'].iloc[i] + ', ' + test_groupped['Country_Region'].iloc[i]

#     location = geolocator.geocode(loc)

#     try:

#         test_groupped['Lat'].iloc[i] = location.latitude

#         test_groupped['Long'].iloc[i] =  location.longitude

#     except:

#         loc = test_groupped['Country_Region'].iloc[i]

#         location = geolocator.geocode(loc)

#         test_groupped['Lat'].iloc[i] = location.latitude

#         test_groupped['Long'].iloc[i] =  location.longitude

        

# bcg_countries = ['Armenia','Azerbaijan','Belarus','Estonia','Georgia','Kazakhstan','Kyrgyzstan','Latvia','Lithuania',

#                  'Moldova','Russia','Tajikistan','Turkmenistan','Ukraine','Uzbekistan',

#                 'United Kingdom','India','Brazil','Germany','Bulgaria','Hungary', 'Poland','Romania','Slovakia',

#                  'Malta','France','Norway','Greece','Singapore','Malaysia', 'Korea, South', 'Taiwan','Japan','Thailand','Sri Lanka','South Africa',

#                 'Philippines','Mongolia','Hong Kong','Pakistan','Ecuador','Argentina','Bolivia','Columbia','Chile','Paraguay','Peru','Uruguay','Venezuela']



# test_groupped['BCG'] = [i in  bcg_countries for i in test_groupped['Country_Region']]

# test = pd.merge(test,test_groupped[['Province_State','Country_Region','Lat','Long','BCG']])



# test.drop('ForecastId',axis=1,inplace=True)



# test.to_csv('test_proc.csv')
train = pd.merge(train,cow,how='left')

test = pd.merge(test,cow,how='left')
train.fillna(0,inplace=True)

test.fillna(0,inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def LbelEnc(df):

    le.fit(df['Province_State'].astype(str))

    df['Province_State'] = le.transform(df['Province_State'].astype(str))

    le.fit(df['Country_Region'].astype(str))

    df['Country_Region'] = le.transform(df['Country_Region'].astype(str))

    le.fit(df['BCG'].astype(str))

    df['BCG'] = le.transform(df['BCG'].astype(str))

    le.fit(df['Region'].astype(str))

    df['Region'] = le.transform(df['Region'].astype(str))

    return df



train = LbelEnc(train)

test = LbelEnc(test)
try:

    test.drop('Unnamed: 0',axis=1,inplace=True)

except:

    1

try:

    train.drop('Unnamed: 0',axis=1,inplace=True)

except:

    1
# import lightgbm as LGB

# from sklearn.model_selection import TimeSeriesSplit,GridSearchCV



# #import sklearn

# #sklearn.metrics.SCORERS.keys()



# cvt = TimeSeriesSplit(n_splits=100) 

# params = {'num_leaves':[31,10,15,20],'max_depth':[-1,15,20],'max_features':['sqrt','log2','auto',None]}

# lgb = LGB.LGBMRegressor(boosting_type='gbdt',learning_rate=1, n_estimators=500)

# model = GridSearchCV(estimator=lgb, param_grid=params,cv=cvt,scoring='neg_root_mean_squared_error') #'neg_median_absolute_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_root_mean_squared_error',
X_train = train[['Province_State', 'Country_Region', 'Date',

       'Lat', 'Long', 'Region',

       'Population', 'Area (sq. mi.)', 'Pop. Density (per sq. mi.)',

        'Climate', 'Birthrate', 'Deathrate']]

test = test[['Province_State', 'Country_Region', 'Date',

       'Lat', 'Long', 'Region',

       'Population', 'Area (sq. mi.)', 'Pop. Density (per sq. mi.)',

        'Climate', 'Birthrate', 'Deathrate']]

Y_train = train[['ConfirmedCases','Fatalities']]

Y_train1 = train['ConfirmedCases']

Y_train2 = train['Fatalities']
# model.fit(X_train,Y_train1);

# print(model.best_params_)

# print('best_score on train : '+str(model.best_score_))
# model.fit(X_train,Y_train2)

# print(model.best_params_)

# print('best_score on train for Fatalities: '+str(model.best_score_))
import numpy as np

from catboost import Pool, CatBoostRegressor



# initialize data

train_data = X_train

train_label = Y_train1

test_data = test



# initialize Pool

train_pool = Pool(train_data, 

                  train_label, 

                  cat_features=[0,1,5])



test_pool = Pool(test_data, 

                 cat_features=[0,1,5]) 



# specify the training parameters 

model = CatBoostRegressor(iterations=10000, 

                          depth=8, 

                          learning_rate=1, 

                          loss_function='RMSE')

#train the model

model.fit(train_pool)



# make the prediction using the resulting model

preds1 = model.predict(test_pool)

print(preds1)
# initialize data

train_data = X_train

train_label = Y_train2

test_data = test



# initialize Pool

train_pool = Pool(train_data, 

                  train_label, 

                  cat_features=[0,1,5])



test_pool = Pool(test_data, 

                 cat_features=[0,1,5]) 



# specify the training parameters 

model = CatBoostRegressor(iterations=10000, 

                          depth=8, 

                          learning_rate=1, 

                          loss_function='RMSE')

#train the model

model.fit(train_pool)



# make the prediction using the resulting model

preds2 = model.predict(test_pool)

print(preds2)
outputFile = pd.DataFrame({"ForecastId": test.index+1,

                           "ConfirmedCases": (np.abs(preds1)+0.5).astype('int'),

                           "Fatalities": (np.abs(preds2)+0.5).astype('int')})
outputFile.to_csv("submission.csv", index=False)