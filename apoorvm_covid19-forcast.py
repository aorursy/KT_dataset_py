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
import lightgbm as lgb

from sklearn import metrics

from sklearn.feature_selection import SelectKBest, f_classif

import category_encoders as ce

from sklearn.experimental import enable_iterative_imputer  

from sklearn.impute import IterativeImputer

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
def Label_encoding(X):

#     cat_feature = [y for y in X.columns if X[y].dtypes == 'object']

    labelencoder = LabelEncoder()

#     print(cat_feature)

    X['Country/Region'] = labelencoder.fit_transform(X['Country/Region'])

    X['Province/State'] = labelencoder.fit_transform(X['Province/State'])

    return X

def iterative_imputer(X):

    numerical_feature = [y for y in X.columns if X[y].dtypes != 'object'

                         and X[y].isnull().sum() != 0]

    numerical_feature = [col for col in numerical_feature if col not in ['ConfirmedCases', 'Fatalities']]

    print(numerical_feature)

    imp_mean = IterativeImputer(max_iter=10, verbose=0)

    X[numerical_feature] = imp_mean.fit_transform(X[numerical_feature])

    

    return X

def train_and_predict(train_X, train_y, valid_X, valid_y, X_test):

#     feature_cols = train.columns.drop('outcome')



    dtrain = lgb.Dataset(train_X, label=train_y)

    dvalid = lgb.Dataset(valid_X, label=valid_y)

    



    

    param = {'num_leaves': 128}

    param['metric'] = 'auc'

    num_round = 1000



    bst = lgb.train(param, dtrain, num_round,

                    valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)

    

#     get predictions

    y_pred = bst.predict(X_test)

    

    return y_pred
#only use on or before 2020-03-11 for public and all for private
covid_train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv",

                          parse_dates=['Date'])
covid_train.head(20)
covid_train.fillna('0', inplace=True)
covid_train['Province/State'].unique()
import datetime
# extracting date from timestamp

covid_train['year'] = pd.DatetimeIndex(covid_train.Date).year

covid_train['month'] = pd.DatetimeIndex(covid_train.Date).month

covid_train['day'] = pd.DatetimeIndex(covid_train.Date).day
# covid_train = covid_train.drop(['new_Date'], axis=1)

# covid_train = covid_train.drop(['Province/State'], axis=1)
covid_train = covid_train[covid_train.Date <= '2020-03-11']
covid_train.info()
covid_train = covid_train.drop(['Date'], axis=1)
covid_train.info()
covid_train['Province/State'].isna().sum()
covid_train.nunique()
covid_train = Label_encoding(covid_train)
covid_train.info()
take_col = [col for col in covid_train if col not in ['ConfirmedCases', 'Fatalities']]

train_X = covid_train[take_col]
train_X.shape
pre_col_1 = [col for col in covid_train if col in ['ConfirmedCases']]

pre_col_2 = [col for col in covid_train if col in ['Fatalities']]
train_y_1 = covid_train[pre_col_1]

train_y_2 = covid_train[pre_col_2]
print(train_X.shape)

print(train_y_1.shape)

print(train_y_2.shape)
# valid_size
valid_fraction = 0.1

valid_size = int(len(covid_train) * valid_fraction)



train_X = train_X[:-2 * valid_size]

confirmed_train_y = train_y_1[:-2 * valid_size]

fatality_train_y = train_y_2[:-2 * valid_size]

valid_X = train_X[-2 * valid_size:]

confirmed_valid_y = train_y_1[-2 * valid_size:]

fatality_valid_y = train_y_2[-2 * valid_size:]
print(train_X.shape)

print(valid_X.shape)

print(confirmed_valid_y.shape)

print(confirmed_train_y.shape)
covid_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv",

                         parse_dates=['Date'])
# extracting date from timestamp

covid_test['year'] = pd.DatetimeIndex(covid_test.Date).year

covid_test['month'] = pd.DatetimeIndex(covid_test.Date).month

covid_test['day'] = pd.DatetimeIndex(covid_test.Date).day

# covid_test = covid_test.drop(['Province/State'], axis=1)

covid_test = covid_test.drop(['Date'], axis=1)
covid_test.info()
covid_test.fillna('0', inplace=True)
covid_test.head(347)
covid_test = Label_encoding(covid_test)
covid_test.info()
confirmer_y_pred = train_and_predict(train_X, confirmed_train_y, valid_X, confirmed_valid_y, covid_test)
fatality_y_pred = train_and_predict(train_X, fatality_train_y, valid_X, fatality_valid_y, covid_test)
confirmer_y_pred = np.round(confirmer_y_pred)
fatality_y_pred = np.round(fatality_y_pred)
pred=pd.DataFrame()

pred['ForecastId']=covid_test['ForecastId']

pred['ConfirmedCases']=confirmer_y_pred

pred['Fatalities']=fatality_y_pred

pred.to_csv('submission.csv',index=False)