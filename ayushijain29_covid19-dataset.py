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

sample_submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

pop = pd.read_csv("/kaggle/input/pop2020/data.csv")

test2=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

train2=pd.read_csv("/kaggle/input/pop2020/data.csv")
pop.head()
pop.rename(columns={'Country or Area':'Country'}, inplace=True)
map_state = {'US':'United States', 

             'Korea, South':'South Korea',

             'Cabo Verde' : 'Carpe Verde',

             'Congo (Brazzaville)': 'Republic of the Congo',

             'Cote d\'Ivoire':'Ivory Coast',

             'Czechia':'Czech Republic',

             'Eswatini':'Swaziland',

             'Holy See':'Vatican City',

          #  'Jersey':'United Kingdom',

             'North Macedonia':'Macedonia',

             'Taiwan*':'Taiwan',

             'occupied Palestinian territory':'Palestine'

            }

map_state_rev = {v: k for k, v in map_state.items()}
pop['country'] = pop['country'].apply(lambda x: map_state_rev[x] if x in map_state_rev else x)

pop.head()
pop.country.unique()
train = pd.merge(train, pop, how='left', left_on = 'Country_Region', right_on = 'country')
train.head()
train.isnull().sum()
train.loc[:, train.isna().any()]
# len(train)

train.head()

# train.Country_Region.nunique() --173

import matplotlib.pyplot as plt

plt.rc('lines', linewidth=2, linestyle='-', marker='*')

plt.rcParams["figure.figsize"] = (25, 8)

plt.plot(train["Country_Region"],train["ConfirmedCases"] ,color='lightblue', linewidth=3)



train.Date.unique()

#from 2020-01-22 till 2020-03-25
sample_submission.head()
#test.head()

#test.ForecastId.nunique()

train.Province_State.unique()

train.tail()
#rename therefor the data columns

train.rename(columns={'Province_State':'Province'}, inplace=True)

train.rename(columns={'Country_Region':'Country'}, inplace=True)

train.rename(columns={'ConfirmedCases':'Confirmed'}, inplace=True)
train
#and we do the same for test set

test.rename(columns={'Province_State':'Province'}, inplace=True)

test.rename(columns={'Country_Region':'Country'}, inplace=True)
train
EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if Province == EMPTY_VAL: return country

    return state





train['Province'].fillna(EMPTY_VAL, inplace=True)

test['Province'].fillna(EMPTY_VAL, inplace=True)
from sklearn.preprocessing import LabelEncoder

# creating initial dataframe

bridge_types = ('Date', 'Province', 'Country', 'Confirmed',

        'Id')

#bridge_types

countries = pd.DataFrame(train, columns=['Country'])

state = pd.DataFrame(train, columns=['Province'])

#countries

# creating instance of labelencoder

labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column

train['Countries'] = labelencoder.fit_transform(train['Country'])

train['State']= labelencoder.fit_transform(train['Province'])

train

# #do the same for test set

test['Countries'] = labelencoder.fit_transform(test['Country'])

test['State']= labelencoder.fit_transform(test['Province'])

#check label encoding 

train['Countries'].head()

train.State.unique()
train


train['Date']= pd.to_datetime(train['Date']) 

test['Date']= pd.to_datetime(test['Date']) 

train = train.set_index(['Date'])

test = test.set_index(['Date'])

train
def create_time_features(df):

    """

    Creates time series features from datetime index

    """

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
create_time_features(train).head()

create_time_features(test).head()
train.head()
train.drop("date", axis=1, inplace=True)

test.drop("date", axis=1, inplace=True)
train
train.isnull().sum()
test
#drop useless columns for train and test set

train.drop(['Country'], axis=1, inplace=True)

# train.drop(['country'], axis=1, inplace=True)

train.drop(['Province'], axis=1, inplace=True)
test.drop(['Country'], axis=1, inplace=True)

test.drop(['Province'], axis=1, inplace=True)
from sklearn.tree import DecisionTreeRegressor  

regressor = DecisionTreeRegressor(random_state = 0) 
# import xgboost as xgb

# from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error



# reg= xgb.XGBRegressor(n_estimators=1000)
train.head()
test
# features that will be used in the model

x = train[['Countries','State','dayofweek','month','dayofyear','weekofyear']]

y1 = train[['Confirmed']]

y2 = train[['Fatalities']]

x_test = test[['Countries','State','dayofweek','month','dayofyear','weekofyear']]
x.head()
#use model on data 

regressor.fit(x,y1)

predict_1 = regressor.predict(x_test)

predict_1 = pd.DataFrame(predict_1)

predict_1.columns = ["Confirmed_predict"]
predict_1.head()
#use model on data 

regressor.fit(x,y2)

predict_2 = regressor.predict(x_test)

predict_2 = pd.DataFrame(predict_2)

predict_2.columns = ["Death_prediction"]

predict_2.head()
Samle_submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")

Samle_submission.columns

submission = Samle_submission[["ForecastId"]]
Final_submission = pd.concat([predict_1,predict_2,submission],axis=1)

Final_submission.head()
Final_submission.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

Final_submission = Final_submission[['ForecastId','ConfirmedCases', 'Fatalities']]



Final_submission["ConfirmedCases"] = Final_submission["ConfirmedCases"].astype(int)

Final_submission["Fatalities"] = Final_submission["Fatalities"].astype(int)
Final_submission.head()
Final_submission.to_csv("submission.csv",index=False)

print('Model ready for submission!')
from sklearn.ensemble import RandomForestRegressor

m1 = RandomForestRegressor(n_estimators=100,max_depth=20)

m1.fit(x,y1)
pred1=m1.predict(x_test)
pred1
m1.fit(x,y2)
pred2=m1.predict(x_test)

pred2
pred1 = pd.DataFrame(pred1)

pred1.columns = ["ConfirmedCases"]
pred2 = pd.DataFrame(pred2)

pred2.columns = ["Fatalities"]
Sample_submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")

Sample_submission.columns

submission = Sample_submission[["ForecastId"]]
submission.head()
RF_submission = pd.concat([pred1,pred2,submission],axis=1)

RF_submission.head()

RF_submission.to_csv("submission.csv",index=False)

print('Model ready for submission!')
import keras


from numpy import array

from keras.models import Sequential

from keras.layers import Dense

# define model

model = Sequential()

model.add(Dense(100, activation='relu', input_dim=5))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(x, y1, epochs=20, verbose=0)

yhat = model.predict(x_test, verbose=0)

print(yhat)
# fit model

model.fit(x, y2, epochs=20, verbose=0)

yhat1 = model.predict(x_test, verbose=0)

print(yhat1)
Sample_submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")

Sample_submission.columns

submission_mlp = Sample_submission[["ForecastId"]]
cases=pd.DataFrame(yhat)

cases.columns=["ConfirmedCases"]

fatal=pd.DataFrame(yhat1)

fatal.columns=["Fatalities"]

Final_submission = pd.concat([cases,fatal,submission_mlp],axis=1)

Final_submission.head()
Final_submission.to_csv("submission.csv",index=False)

print('Model ready for submission!')
import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error

reg = xgb.XGBRegressor(n_estimators=1000)

reg.fit(x, y1,

       verbose=False)
reg = xgb.XGBRegressor(n_estimators=1000)

reg.fit(x, y2,

       verbose=False)
XG_fatalities = reg.predict(x_test)
XG_Confirmed = reg.predict(x_test)

XG_Confirmed
XG_fatalities
Sample_submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")

Sample_submission.columns

submission_xg = Sample_submission[["ForecastId"]]
cases=pd.DataFrame(XG_Confirmed)

cases.columns=["ConfirmedCases"]

fatal=pd.DataFrame(XG_fatalities)

fatal.columns=["Fatalities"]

Final_submission = pd.concat([cases,fatal,submission_xg],axis=1)

Final_submission.head()
from sklearn.model_selection import GridSearchCV

import time

param_grid = {'n_estimators': [1000]}
def gridSearchCV(model, X_Train, y_Train, param_grid, cv=10, scoring='neg_mean_squared_error'):

    start = time.time()

    

    grid_cv = GridSearchCV(model, param_grid, cv=10, scoring="neg_mean_squared_error")

    grid_cv.fit(X_Train, y_Train)

    

    print (f'{type(model).__name__} Hyper Paramter Tuning took a Time: {time.time() - start}')

    print (f'Best {scoring}: {grid_cv.best_score_}')

    print ("Best Hyper Parameters:\n{}".format(grid_cv.best_params_))

    

    return grid_cv.best_estimator_
from xgboost import XGBRegressor



model = XGBRegressor()



model1 = gridSearchCV(model, x, y1, param_grid, 10, 'neg_mean_squared_error')

model2 = gridSearchCV(model, x, y2, param_grid, 10, 'neg_mean_squared_error')
y1_pred = model1.predict(x_test)

y1_pred = y1_pred.round()
y2_pred = model2.predict(x_test)

y2_pred = y2_pred.round()
Sample_submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")

Sample_submission.columns

submission_xg = Sample_submission[["ForecastId"]]
cases=pd.DataFrame(y1_pred)

cases.columns=["ConfirmedCases"]

fatal=pd.DataFrame(y2_pred)

fatal.columns=["Fatalities"]

Final_submission = pd.concat([cases,fatal,submission_xg],axis=1)

Final_submission.head(20)