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
from datetime import datetime

import time



import numpy as np

import pandas as pd

from sklearn import preprocessing

import xgboost as xgb

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor

import xgboost

import math



from scipy.stats import pearsonr

from sklearn.linear_model import LinearRegression

from sklearn import tree, linear_model

from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_squared_error, r2_score

from math import sqrt
train= pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

test= pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
train.head()
test.head()
#print(train.groupby(['Date','Lat']).count().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'])
display(train.head(5))

display(train.describe())

print("Number of Country_Region: ", train['Country_Region'].nunique())

print("Number of Country_Region: ", train['Country_Region'].unique())

print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")

print("Countries with Province/State informed: ", train[train['Province_State'].isna()==False]['Country_Region'].unique())

print("Countries with Province/State informed: ", train[train['Province_State'].isna()==False]['Province_State'].unique())

print(train.groupby(['Date']).mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'][[1,2]])

print(train.groupby('Date').mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'][[4,5,6]])
print(train.groupby(['Date','Country_Region']).mean().sort_values(by='ConfirmedCases', ascending=False)['ConfirmedCases'])
training_data=train.groupby('Date')['ConfirmedCases','Fatalities'].sum().reset_index()
training_data=train.groupby('Date')['ConfirmedCases','Fatalities'].sum().reset_index()
training_data
y_train = train[["ConfirmedCases", "Fatalities"]]

train = train[["Province_State","Country_Region","Date"]]

X_test_Id = test.loc[:, 'ForecastId']

test = test[["Province_State","Country_Region","Date"]]
#print(train_encoded.count())

print(y_train)

#xyz = train.groupby(['Country_Region']).count().sort_values(by='ConfirmedCases', ascending=False)()

#train.groupby(['Country_Region']).sort_values(by='Date', ascending=False)[:100]
#train.groupby(['Country_Region','col2']).count()
#train.groupby(['Date','Country_Region'])['count','ConfirmedCases']
#train = train.sort_values(by=['Date','Country_Region'], ascending=True)
#train[train.groupby(['Date','Country_Region'])'count','ConfirmedCases')]
#train['count']=1
#train['count'] = train.groupby(['Date','Country_Region'])['ConfirmedCases'].apply()#
EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
from sklearn.preprocessing import LabelEncoder



print("fill blanks and add region for counting")

#train.fillna(' ',inplace=True)

#train['Lat']=train['Province_State']+train['Country_Region']

#train.drop('Province_State',axis=1,inplace=True)

#train.drop('Country_Region',axis=1,inplace=True)





cols = ['ConfirmedCases', 'Fatalities']

index_split = train.shape[0]



full_df = pd.concat([train,test],sort=False)

full_df.fillna(' ',inplace=True)

#full_df['Lat']=full_df['Province_State']+full_df['Country_Region']

#X_xTrain['State'].fillna(EMPTY_VAL, inplace=True)

full_df['Province_State'] = full_df.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)







#full_df.drop('Province_State',axis=1,inplace=True)

#full_df.drop('Country_Region',axis=1,inplace=True)

display(full_df.head())

#full_df = pd.concat([train.drop(cols, axis=1))

#full_df.Date = full_df.Date.astype('int64')

full_df['Mon'] = full_df['Date'].apply(lambda x: int(x.split('-')[1]))

full_df['Day'] = full_df['Date'].apply(lambda x: int(x.split('-')[2]))

full_df['serial'] = full_df['Mon'] * 30 + full_df['Day']

full_df['serial'] = full_df['serial'] - full_df['serial'].min()

full_df.Date = pd.to_datetime(full_df.Date)

full_df['Date'] = full_df['Date'].apply(pd.to_datetime)



full_df['day_of_week'] = full_df['Date'].apply(lambda ts: ts.weekday()).astype('int')

full_df['month'] = full_df['Date'].apply(lambda ts: ts.month)

full_df['day'] = full_df['Date'].apply(lambda ts: ts.day)

full_df.loc[:, 'Date'] = full_df.Date.dt.strftime("%m%d")

full_df['Date']  = full_df['Date'].astype(int)



#full_df.drop(['Province_State','Country_Region'],axis=1, inplace= True )

full_df.drop(['Mon','Day'],axis=1, inplace= True )

#full_df.drop(['Date', 'Province_State','Country_Region','Mon','Day'],axis=1, inplace= True )

display(full_df.dtypes)

display(full_df.head())

le = LabelEncoder()

def CustomLabelEncoder(df):

    for c in df.columns:

        if df.dtypes[c] == object:

            le.fit(df[c].astype(str))

            df[c] = le.transform(df[c].astype(str))

    return df


full_df_encoded = CustomLabelEncoder(full_df)



train_encoded = full_df[:index_split]

test_encoded= full_df[index_split:]

train_encoded.count()
test_encoded.iloc[250:350,:]
train_encoded.tail()
test_encoded.head()
train.head()
full_df_encoded.head(10)
train_encoded.info()
test_encoded
from sklearn.model_selection import train_test_split



X_train1, X_test1, y_train1, y_test1 = train_test_split(train_encoded, y_train[['ConfirmedCases']],test_size=0.2, random_state=48)



X_train2, X_test2, y_train2, y_test2 = train_test_split(train_encoded, y_train[['Fatalities']] ,test_size=0.2, random_state=48)

X_train1
y_train1,y_test1,y_train2,y_test2
# features that will be used in the model

y1 = y_train[['ConfirmedCases']]

y2 = y_train[['Fatalities']]
from sklearn.tree import DecisionTreeRegressor  

regressor = DecisionTreeRegressor(random_state = 0) 

from sklearn.metrics import mean_squared_error, mean_absolute_error

regressor.fit(X_train1,y_train1)

#predict_dt1 = regressor.predict(X_test1)

#predict_dt1 = pd.DataFrame(predict_dt1)

#predict_dt1.columns = ["ConfirmedCases"]

#print(predict_dt1)

display(regressor.score(X_test1,y_test1))



predict_dt1 = regressor.predict(X_test1)

#predict_dt1 = pd.DataFrame(predict_dt1)

#predict_dt1.columns = ["ConfirmedCases"]

predict_dt1 = predict_dt1.astype(int)

predict_dt1
print(regressor.score(X_test1,y_test1)) 

print(explained_variance_score(predict_dt1,y_test1)) 

print(mean_squared_error(y_test1,predict_dt1)) 

print(r2_score(y_test1,predict_dt1)) 

mae = mean_absolute_error(y_test1, predict_dt1) 

print('MAE: %f' % mae) 

mse = mean_squared_error(y_test1, predict_dt1) 

rmse = sqrt(mse) 

print('RMSE: %f' % rmse)
regressor.fit(train_encoded, y_train[['ConfirmedCases']])

#predict_dt1 = regressor.predict(X_test1)

#predict_dt1 = pd.DataFrame(predict_dt1)

#predict_dt1.columns = ["ConfirmedCases"]

#print(predict_dt1)

print(regressor.score(X_test1,y_test1))



predict_dt1 = regressor.predict(test_encoded)

#predict_dt1 = pd.DataFrame(predict_dt1)

#predict_dt1.columns = ["ConfirmedCases"]

predict_dt1 = predict_dt1.astype(int)

predict_dt1
regressor.fit(train_encoded, y_train[['Fatalities']])

#predict_dt2 = regressor.predict(X_test2)

#predict_dt2 =predict_dt1.astype(int)

#predict_dt2 = pd.DataFrame(predict_dt2)

#predict_dt2.columns = ["Fatalities"]

predict_dt2 = regressor.predict(test_encoded)

#predict_dt1 = pd.DataFrame(predict_dt1)

#predict_dt1.columns = ["ConfirmedCases"]

predict_dt2 = predict_dt2.astype(int)

predict_dt2
print(X_test_Id,predict_dt1,predict_dt2)
sub = pd.DataFrame({'ForecastId':X_test_Id,'ConfirmedCases': predict_dt1, 'Fatalities': predict_dt2})



sub.ForecastId = sub.ForecastId.astype('int')

sub.head()

sub.to_csv('submission.csv', index=False)
import lightgbm as lgb

SEED = 42

params = {'num_leaves': 8,

          'min_data_in_leaf': 5,  # 42,

          'objective': 'regression',

          'max_depth': 8,

          'learning_rate': 0.02,

          'boosting': 'gbdt',

          'bagging_freq': 5,  # 5

          'bagging_fraction': 0.8,  # 0.5,

          'feature_fraction': 0.8201,

          'bagging_seed': SEED,

          'reg_alpha': 1,  # 1.728910519108444,

          'reg_lambda': 4.9847051755586085,

          'random_state': SEED,

          'metric': 'mse',

          'verbosity': 100,

          'min_gain_to_split': 0.02,  # 0.01077313523861969,

          'min_child_weight': 5,  # 19.428902804238373,

          'num_threads': 6,

          }
lgb_train = lgb.Dataset(X_train1, y_train1) 

lgb_eval = lgb.Dataset(X_test1, y_test1, reference=lgb_train)



#specify your configurations as a dict

params = { 'boosting_type': 'gbdt', 'objective': 'regression', 'metric': {'l2', 'l1'}, 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': 0 }



print('Starting training...')



#train

gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
#model.predict(X_test1, num_iteration=best_itr)
y_test1
#sub.head()
xgb = xgboost.XGBRegressor(n_estimators=1000)
xgb1=xgb.fit(X_train1,y_train1)

predictions = xgb1.predict(X_test1)

print(explained_variance_score(predictions,y_test1))



predict_xgb1 = xgb1.predict(test_encoded)

#predict_xg2 = pd.DataFrame(predict_dt2)

#predict_xg2.columns = ["Fatalities"]

print(xgb1.score(X_test1,y_test1))

print(explained_variance_score(predictions,y_test1))

print(mean_squared_error(y_test1,predictions))

print(r2_score(y_test1,predictions))
#xgb1=xgb.fit(X_train1,y_train1)

predictions_xg = xgb1.predict(test_encoded)

#print(explained_variance_score(predictions,y_test1))

print(predictions_xg)
print(explained_variance_score(predictions_xg,predict_dt1))
xgb1.score(X_test1,y_test1)
xgb2=xgb.fit(X_train2,y_train2)

predictions = xgb2.predict(X_test2)

print(explained_variance_score(predictions,y_test2))



predict_xgb2 = xgb2.predict(test_encoded)

#predict_xg2 = pd.DataFrame(predict_dt2)

#predict_xg2.columns = ["Fatalities"]

print(xgb2.score(X_test2,y_test2))

print(explained_variance_score(predictions,y_test2))

print(mean_squared_error(y_test2,predictions))

print(r2_score(y_test2,predictions))
xgb1=xgb.fit(X_train1,y_train1)

predictions = xgb1.predict(X_test1)

print(explained_variance_score(predictions,y_test1))



predict_xgb2 = xgb1.predict(test_encoded)

#predict_xg2 = pd.DataFrame(predict_dt2)

#predict_xg2.columns = ["Fatalities"]

print(xgb1.score(X_test1,y_test1))

print(explained_variance_score(predictions,y_test1))

print(mean_squared_error(y_test1,predictions))

print(r2_score(y_test1,predictions))
#predict_dt1 = regressor.predict(X_test1)

#predict_dt1 = pd.DataFrame(predict_dt1)

#predict_dt1.columns = ["ConfirmedCases"]

#print(predict_dt1)

display(xgb.score(X_test1,y_test1))



predict_dt1 = xgb.predict(test_encoded)

#predict_dt1 = pd.DataFrame(predict_dt1)

#predict_dt1.columns = ["ConfirmedCases"]

predict_dt1 = predict_dt1.astype(int)

predict_dt1
xgb.fit(train_encoded, y_train[['ConfirmedCases']])

#predict_dt1 = regressor.predict(X_test1)

#predict_dt1 = pd.DataFrame(predict_dt1)

#predict_dt1.columns = ["ConfirmedCases"]

#print(predict_dt1)

print(xgb.score(X_test1,y_test1))



predict_dt1 = xgb.predict(test_encoded)

#predict_dt1 = pd.DataFrame(predict_dt1)

#predict_dt1.columns = ["ConfirmedCases"]

predict_dt1 = predict_dt1.astype(int)

predict_dt1
xgb1=xgb.fit(X_train2,y_train2)

predictions = xgb1.predict(X_test2)

print(explained_variance_score(predictions,y_test2))



predict_xgb2 = xgb1.predict(test_encoded)

#predict_xg2 = pd.DataFrame(predict_dt2)

#predict_xg2.columns = ["Fatalities"]

print(xgb1.score(X_test2,y_test2))

print(explained_variance_score(predictions,y_test2))

print(mean_squared_error(y_test2,predictions))

print(r2_score(y_test2,predictions))
#predict_dt1 = regressor.predict(X_test1)

#predict_dt1 = pd.DataFrame(predict_dt1)

#predict_dt1.columns = ["ConfirmedCases"]

#print(predict_dt1)

display(xgb.score(X_test2,y_test2))



predict_dt2 = xgb.predict(test_encoded)

#predict_dt1 = pd.DataFrame(predict_dt1)

#predict_dt1.columns = ["ConfirmedCases"]

predict_dt2 = predict_dt1.astype(int)

predict_dt2
xgb.fit(train_encoded, y_train[['Fatalities']])

#predict_dt2 = regressor.predict(X_test2)

#predict_dt2 =predict_dt1.astype(int)

#predict_dt2 = pd.DataFrame(predict_dt2)

#predict_dt2.columns = ["Fatalities"]

predict_dt2 = xgb.predict(test_encoded)

#predict_dt1 = pd.DataFrame(predict_dt1)

#predict_dt1.columns = ["ConfirmedCases"]

predict_dt2 = predict_dt2.astype(int)

predict_dt2