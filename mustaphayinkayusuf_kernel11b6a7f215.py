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

from matplotlib import pyplot as plt

import numpy as np

%matplotlib inline
train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

train.head()
test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")

test.head()
len(train['Country/Region'].unique().sum())
train['Country/Region'].unique()
grouped = train.groupby('Country/Region')

grouped.groups
print(train.ConfirmedCases.sum())

print(train.Fatalities.sum())
import datetime as dt

train['Dates'] = pd.to_datetime(train.Date)



train['Year'] = train.Dates.dt.year

train['Month'] = train.Dates.dt.month

train['Week'] = train.Dates.dt.week

train['Name of month'] = train.Dates.dt.month_name()

train['Week_day_name'] = train.Dates.dt.weekday_name

train['Day_of_year'] = train.Dates.dt.dayofyear
target = train['ConfirmedCases'].values

features = train[['Month', 'Week', 'Day_of_year']].values

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 4  )
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

log.fit(x_train, y_train)

log.score(x_test, y_test)


test.head()
import datetime as dt

test['Dates'] = pd.to_datetime(test.Date)



test['Year'] = test.Dates.dt.year

test['Month'] = test.Dates.dt.month

test['Week'] = test.Dates.dt.week

test['Name of month'] = test.Dates.dt.month_name()

test['Week_day_name'] = test.Dates.dt.weekday_name

test['Day_of_year'] = test.Dates.dt.dayofyear
features1 = test[['Month', 'Week', 'Day_of_year']].values
pred1 = log.predict(features1)

pred1
target = train['Fatalities'].values

features = train[['Month', 'Week', 'Day_of_year']].values

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 4  )
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

log.fit(x_train, y_train)

log.score(x_test, y_test)
features2 = test[['Month', 'Week', 'Day_of_year']].values

pred2 = log.predict(features2)
len(pred2)
sub = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")

sub.head()
sub['ConfirmedCases'] = pred1

sub['Fatalities'] = pred2
sub.head()
sub1 = sub.to_csv('submission.csv', index = False)