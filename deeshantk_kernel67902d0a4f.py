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
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])

train['dayofyear'] = train['Date'].dt.dayofyear

test['dayofyear'] = test['Date'].dt.dayofyear

train['month'] = pd.DatetimeIndex(train['Date']).month

test['month'] = pd.DatetimeIndex(test['Date']).month
train['Province_State'].fillna('None', inplace = True)

test['Province_State'].fillna('None', inplace = True)

from sklearn.preprocessing import LabelEncoder



label = LabelEncoder()



train['Countries'] = label.fit_transform(train['Country_Region'])

test['Countries'] = label.fit_transform(test['Country_Region'])

train['State'] = label.fit_transform(train['Province_State'])

test['State'] = label.fit_transform(test['Province_State'])

y = train['ConfirmedCases']

y1 = train['Fatalities']
def drop(df, d):

    df.drop(d, axis = 1, inplace = True)
drop(train, 'Id')

drop(train, 'Province_State')

drop(train, 'Country_Region')

drop(train, 'Date')

drop(train, 'ConfirmedCases')

drop(train, 'Fatalities')

drop(test, 'Province_State')

drop(test, 'ForecastId')

drop(test, 'Country_Region')

drop(test, 'Date')
train.head()
test.head()
from sklearn.tree import DecisionTreeRegressor



model = DecisionTreeRegressor(random_state = 1)
model.fit(train, y)
ConfirmedCases = model.predict(test)

ConfirmedCases = ConfirmedCases.astype(int)

test['ConfirmedCases'] = ConfirmedCases

train['ConfirmedCases'] = y
model.fit(train, y1)
Fatalities = model.predict(test)
Fatalities = Fatalities.astype(int)
df = pd.DataFrame({'ForecastId': range(1, 13460)})

df['ConfirmedCases'] = ConfirmedCases

df['Fatalities'] = Fatalities

df.to_csv('submission.csv', index = False)