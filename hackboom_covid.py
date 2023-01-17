# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')

test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
from datetime import datetime

train["Date"] = pd.to_datetime(train["Date"])

test["Date"] = pd.to_datetime(test["Date"])
train['unique_region'] = train['Country_Region']

train['unique_region'][train['Province_State'].isna() == False] = train['Province_State'] + ', ' + train['Country_Region']

test['unique_region'] = test['Country_Region']

test['unique_region'][test['Province_State'].isna() == False] = test['Province_State'] + ', ' + test['Country_Region']
train['month'] = train['Date'].dt.month

train['day'] = train['Date'].dt.day

test['month'] = test['Date'].dt.month

test['day'] = test['Date'].dt.day
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import MinMaxScaler

poly_con = PolynomialFeatures(4)

poly_fatal = PolynomialFeatures(4)

lr_con = LinearRegression()

lr_fatal = LinearRegression()

submission = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})
for region in train['unique_region'].unique():

    train_df = train[train['unique_region']== region]

    y_train_con = train_df['ConfirmedCases'].values

    y_train_fatal = train_df['Fatalities'].values

    

    x_train = train_df[['month','day']]

    

    test_df = test[test['unique_region']== region]

    x_test = test_df[['month','day']]

    ForecastId = test_df["ForecastId"].values

    

    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test)

    

    scaler_con = MinMaxScaler()

    scaler_fatal = MinMaxScaler()

    

    y_train_con=scaler_con.fit_transform(y_train_con.reshape(-1, 1))

    y_train_fatal=scaler_fatal.fit_transform(y_train_fatal.reshape(-1, 1))

    

    #확진자

    x_train_poly = poly_con.fit_transform(x_train)

    lr_con.fit(x_train_poly, y_train_con)

    

    x_test_poly = poly_con.fit_transform(x_test)

    test_con = lr_con.predict(x_test_poly)

    test_con = scaler_con.inverse_transform(test_con).flatten()

    

    x_train_poly = poly_fatal.fit_transform(x_train)

    lr_fatal.fit(x_train_poly, y_train_fatal)

    

    x_test_poly = poly_fatal.fit_transform(x_test)

    test_fatal = lr_fatal.predict(x_test_poly)

    test_fatal = scaler_fatal.inverse_transform(test_fatal).flatten()

    

    result = pd.DataFrame({'ForecastId': ForecastId, 'ConfirmedCases': test_con, 'Fatalities': test_fatal})

    

    submission = pd.concat([submission, result])
submission['ForecastId'] = submission['ForecastId'].astype('int32')

submission['ConfirmedCases'] = submission['ConfirmedCases'].astype('int32')

submission['Fatalities'] = submission['Fatalities'].astype('int32')
submission.to_csv('submission.csv', index = False)