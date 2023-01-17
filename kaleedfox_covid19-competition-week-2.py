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
training = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

testing = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

data_ = pd.read_csv("/kaggle/input/covid19-demographic-predictors/covid19_by_country.csv")
data_
training
#change the name of the 'country' feature to match 'Country_Region' on the train set 

data_['Country_Region']= data_.Country

data_.drop('Country',axis=1,  inplace =True)
training.info()
print(data_.shape)

print(training.shape)
#missing values

training.isnull().sum()
#missing values

data_.isnull().sum()
data_['Quarantine_date'] = pd.to_datetime(data_.Quarantine)

data_['Restrictions_date'] = pd.to_datetime(data_.Restrictions)

data_['Schools_date'] = pd.to_datetime(data_.Schools)

data_.drop(['Schools', 'Restrictions', 'Quarantine'], axis =1, inplace = True)
training.Date = pd.to_datetime(training.Date)
training = training.fillna({'Province_State': 'Unknown'})

testing = testing.fillna({'Province_State': 'Unknown'})
data_.info()
training.info()
df = training.groupby(['Country_Region', 'Date'], as_index=False).sum()

df_test = testing.groupby(['Country_Region', 'Date'], as_index=False).sum()
df_test[df_test.Country_Region == 'Italy']
df[df.Country_Region == 'Italy']
len(df.Country_Region.unique())
len(df_test.Country_Region.unique())
train = pd.merge(training, data_, on=['Country_Region'], how= 'left')

test  = pd.merge(testing, data_, on=['Country_Region'], how= 'left')
train.shape
train.isna().sum()
test.shape
data_[data_.Restrictions_date.notnull()][['Country_Region', 'Quarantine_date']]
train.loc[(train['Date'] == '2020-03-20') &(train.Country_Region == 'Argentina') ]
test['Quarantine'] = 0

test['Schools'] = 0

test['Restrictions'] = 0



test.loc[(test.Country_Region == 'Argentina'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'Austria'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'Belgium'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'China'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'Colombia'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'Denmark'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'France'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'Germany'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'India'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'Israel'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'Italy'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'Malaysia'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'New Zealand'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'Peru'), 'Quarantine' ] = 1

test.loc[(test.Country_Region == 'Spain'), 'Quarantine' ] = 1



test.loc[(test.Country_Region == 'Israel'), 'Schools' ] = 1



test.loc[(test.Country_Region == 'Israel'), 'Restrictions' ] = 1



test.drop(['Quarantine_date', 'Schools_date', 'Restrictions_date'], axis = 1, inplace = True)
train['Quarantine'] = 0

train['Schools'] = 0

train['Restrictions'] = 0



train.loc[(train['Date'] >= '2020-03-20') &(train.Country_Region == 'Argentina'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-16') &(train.Country_Region == 'Austria'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-18') &(train.Country_Region == 'Belgium'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-01-24') &(train.Country_Region == 'China'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-25') &(train.Country_Region == 'Colombia'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-16') &(train.Country_Region == 'Denmark'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-17') &(train.Country_Region == 'France'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-21') &(train.Country_Region == 'Germany'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-23') &(train.Country_Region == 'India'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-19') &(train.Country_Region == 'Israel'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-08') &(train.Country_Region == 'Italy'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-18') &(train.Country_Region == 'Malaysia'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-23') &(train.Country_Region == 'New Zealand'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-15') &(train.Country_Region == 'Peru'), 'Quarantine' ] = 1

train.loc[(train['Date'] >= '2020-03-15') &(train.Country_Region == 'Spain'), 'Quarantine' ] = 1



train.loc[(train['Date'] >= '2020-03-19') &(train.Country_Region == 'Israel'), 'Schools' ] = 1



train.loc[(train['Date'] >= '2020-03-19') &(train.Country_Region == 'Israel'), 'Restrictions' ] = 1



train.drop(['Quarantine_date', 'Schools_date', 'Restrictions_date'], axis = 1, inplace = True)
train[train.Quarantine == 1][['Country_Region', 'Date']].head(50)
data_[data_.Quarantine_date.notnull()]
train['Quarantine'].any() ==1
train[train.Country_Region=='Italy']
train[train['Restrictions'] == 1]
train.columns
train.hist(figsize=(11,10))
train.drop(['Tests','Test Pop', 'Density', 'Urban Pop', 'sex0', 'sex14',

            'sex25', 'sex54', 'sex64', 'sex65plus', 'Sex Ratio', 'lung',

            'Female Lung', 'Male Lung', 'Crime Index', 'Population 2020',

            'Smoking 2016', 'Females 2018', 'Total Infected','Total Deaths',

            'Total Recovered', 'Hospital Bed', 'Median Age', 'GDP 2018'], axis = 1, inplace = True)
test.drop(['Tests','Test Pop', 'Density', 'Urban Pop', 'sex0', 'sex14',

           'sex25', 'sex54', 'sex64', 'sex65plus', 'Sex Ratio', 'lung',

           'Female Lung', 'Male Lung', 'Crime Index', 'Population 2020',

           'Smoking 2016', 'Females 2018', 'Total Infected', 'Total Deaths',

           'Total Recovered', 'Hospital Bed', 'Median Age', 'GDP 2018'], axis = 1, inplace = True)
print(train.describe())
print(test.describe())
train.isna().sum()
test.isna().sum()
test.Date = pd.to_datetime(test.Date)
test.info()
def create_time_features(df):

    """

    Creates time series features from datetime index

    """

    df['date'] = df.index

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['quarter'] = df['Date'].dt.quarter

    df['month'] = df['Date'].dt.month

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['dayofmonth'] = df['Date'].dt.day

    df['weekofyear'] = df['Date'].dt.weekofyear

    

    X = df[['dayofweek','quarter','month',

           'dayofyear','dayofmonth','weekofyear']]

    return X
train.columns
train.Date = pd.to_datetime(train.Date)
create_time_features(train)

create_time_features(test)
train.columns
train.drop(["Id","Date", 'date'], axis=1, inplace=True)

test.drop(["Date", 'date'], axis=1, inplace=True)
train = pd.concat([train,pd.get_dummies(train['Country_Region'], prefix='ps')],axis=1)

train.drop(['Country_Region'],axis=1, inplace=True)



train = pd.concat([train,pd.get_dummies(train['Province_State'], prefix='ps')],axis=1)

train.drop(['Province_State'],axis=1, inplace=True)
test = pd.concat([test,pd.get_dummies(test['Country_Region'], prefix='ps')],axis=1)

test.drop(['Country_Region'],axis=1, inplace=True)



test = pd.concat([test,pd.get_dummies(test['Province_State'], prefix='ps')],axis=1)

test.drop(['Province_State'],axis=1, inplace=True)
train.shape
train = train.loc[:,~train.columns.duplicated()]

test = test.loc[:,~test.columns.duplicated()]
y = train.Fatalities

x = train.drop(["Fatalities", "ConfirmedCases"], axis = 1)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from catboost import CatBoostRegressor

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

%matplotlib inline
from sklearn.model_selection import train_test_split





def linear_models(X_model, y_model, model_name):



    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, random_state = 42)

    if model_name == "ridge":

        model = linear_model.Ridge(alpha = 0.1, random_state = 42).fit(X_train, y_train)

    if model_name == "lasso":

        model = linear_model.Lasso(alpha = 0.5, random_state = 42).fit(X_train, y_train)

    if model_name == "xgb":

        model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 0.1, n_estimators = 10, random_state = 42).fit(X_train, y_train)

        print(model.get_booster().get_score(importance_type="gain"))

    if model_name == "catboost":

        model = CatBoostRegressor(num_leaves = 31,random_state = 42,learning_rate = 0.05).fit(X_train, y_train)

    predictions = model.predict(X_test)

    return predictions, y_test
def error_metrics(model, predictions, y_test):

    print("Model: ", model)

    # The mean squared error

    print("--Mean squared error: %.2f" % mean_squared_error(y_test, predictions))

    # RMS

    print('--Root Mean Squared Error: %.2f' % np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    # Explained variance score: 1 is perfect prediction

    print('--Variance score: %.2f' % r2_score(y_test, predictions))
# Take a look at some of the results

def inspect_df(predictions, y_test):

    true_vs_pred = np.vstack((predictions, y_test))

    true_df = pd.DataFrame(true_vs_pred)

    true_df = true_df.transpose()

    true_df.columns = ["Predicted", "Actual"]

    return true_df
from IPython.display import display_html

def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
ridge_pred, y_test = linear_models(x, y, "ridge")

lasso_pred, y_test = linear_models(x, y, "lasso")

xgb_pred, y_test   = linear_models(x, y, "xgb")

lgb_pred, y_test   = linear_models(x, y, "catboost")
error_metrics("Ridge", ridge_pred, y_test)

error_metrics("Lasso", lasso_pred, y_test)

error_metrics("xgboost regression", xgb_pred, y_test)

error_metrics("catboost regression", lgb_pred, y_test)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 42)



model = CatBoostRegressor(random_state = 42)



"""grid = {'learning_rate': [0.03, 0.1],

        'depth': [4, 6, 10],

        'l2_leaf_reg': [1, 3, 5, 7, 9],

        #'num_leaves' : [5, 15, 31, 60],

        'bagging_temperature' : [0, 1, 5, 10]}



grid_search_model = model.grid_search(grid,

                                      X=X_train, 

                                      y=y_train,

                                      cv = 5

                                        )"""

model = CatBoostRegressor(num_leaves = 31, bagging_temperature= 0, depth = 6, l2_leaf_reg = 1, learning_rate = 0.03).fit(X_train, y_train)

predictions = model.predict(X_test)

error_metrics("Catboost regression", predictions, y_test)
print(model.get_feature_importance(prettified = True))
test.columns
submit = pd.DataFrame()

submit['ForecastId'] = test['ForecastId']

test.drop(["ForecastId"], axis = 1, inplace = True)
submit
fatilities = model.predict(test)
y = train.ConfirmedCases

x = train.drop(["Fatalities", "ConfirmedCases"], axis = 1)
ridge_pred, y_test = linear_models(x, y, "ridge")

lasso_pred, y_test = linear_models(x, y, "lasso")

xgb_pred, y_test   = linear_models(x, y, "xgb")

lgb_pred, y_test   = linear_models(x, y, "catboost")
error_metrics("Ridge", ridge_pred, y_test)

error_metrics("Lasso", lasso_pred, y_test)

error_metrics("xgboost regression", xgb_pred, y_test)

error_metrics("catboost regression", lgb_pred, y_test)
print(model.get_feature_importance(prettified = True))
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 42)



model = CatBoostRegressor(random_state = 42)



"""grid = {'learning_rate': [0.03, 0.1],

        'depth': [4, 6, 10],

        'l2_leaf_reg': [1, 3, 5, 7, 9],

        #'num_leaves' : [5, 15, 31, 60],

        'bagging_temperature' : [0, 1, 5, 10]}



grid_search_model = model.grid_search(grid,

                                      X=X_train, 

                                      y=y_train,

                                      cv = 5

                                        )"""

model = CatBoostRegressor(num_leaves = 31, bagging_temperature= 0, depth = 4, l2_leaf_reg = 5, learning_rate = 0.1).fit(X_train, y_train)

predictions = model.predict(X_test)

error_metrics("Catboost regression", predictions, y_test)
print(model.get_feature_importance(prettified = True))
confirmedCases = model.predict(test)
submit['ConfirmedCases'] = confirmedCases

submit['Fatalities'] = fatilities
submit
submit.to_csv('submission.csv',index=False)