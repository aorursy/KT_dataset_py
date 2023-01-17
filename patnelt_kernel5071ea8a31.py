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
# save filepath to variable for easier access

train_file_path = '/kaggle/input/covid19-global-forecasting-week-4/train.csv'

test_file_path = '/kaggle/input/covid19-global-forecasting-week-4/test.csv'

submission_file_path = '/kaggle/input/covid19-global-forecasting-week-4/submission.csv'

# read the data and store data in DataFrame titled melbourne_data

train_df = pd.read_csv(train_file_path)

test_df = pd.read_csv(test_file_path)

submission = pd.read_csv(submission_file_path)

# print a summary of the data in Melbourne data

train_df.head()
test_df.head()
submission.head()
train_df.columns
train_df.isna().sum()
test_df.isna().sum()
train_df['Province_State'].fillna("",inplace = True)

test_df['Province_State'].fillna("",inplace = True)
train_df['Country_Region'] = train_df['Country_Region'] + '-' + train_df['Province_State']

test_df['Country_Region'] = test_df['Country_Region'] + '-' + test_df['Province_State']

del train_df['Province_State']

del test_df['Province_State']

train_df.head()
test_df.head()
train_df['ConfirmedCases'] = train_df['ConfirmedCases'].apply(int)

train_df['Fatalities'] = train_df['Fatalities'].apply(int)

cases = train_df.ConfirmedCases

fatalities = train_df.Fatalities
covid_features = ['Country_Region', 'Date']

X = train_df[covid_features]

X.head()
del test_df['ForecastId']

test_df.head()
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

X['Country_Region'] = lb.fit_transform(X['Country_Region'])

test_df['Country_Region'] = lb.transform(test_df['Country_Region'])

X.head()
test_df.head()
# Get names of columns with missing values

cols_with_missing = [col for col in X.columns

                     if X[col].isnull().any()]

cols_with_missing
# Get names of columns with missing values

cols_with_missing = [col for col in test_df.columns

                     if test_df[col].isnull().any()]

cols_with_missing
month = []

day = []



def transform_date(X):

  for i in X.Date:

    y, m, d = i.split('-')

    m = int(m)

    d = int(d)

    month.append(m)

    day.append(d)

    

transform_date(X)



X['Month'] = month

X['Day'] = day

del X['Date']

X.head()
month = []

day = []

transform_date(test_df)



test_df['Month'] = month

test_df['Day'] = day

del test_df['Date']

test_df.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(X.values)

x_test = scaler.transform(test_df.values)
from xgboost import XGBRegressor

#my_model = XGBRegressor(n_estimators = 1000 , random_state = 0 , max_depth = 26)

my_model = XGBRegressor(n_estimators = 2500 , random_state = 0 , max_depth = 27)

my_model.fit(x_train,cases)
import numpy as np

cases_pred = my_model.predict(x_test)

cases_pred = np.around(cases_pred,decimals = 0)

type(cases_pred)
submission['ConfirmedCases'] = cases_pred
my_model.fit(x_train,fatalities)
fatalities_pred = my_model.predict(x_test)

fatalities_pred = np.around(fatalities_pred,decimals = 0)

fatalities_pred
submission['Fatalities'] = fatalities_pred
submission.head()
submission.to_csv("submission.csv" , index = False)