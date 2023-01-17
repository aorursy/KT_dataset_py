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
train = pd.read_csv( '../input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv( '../input/covid19-global-forecasting-week-4/test.csv')

submission = pd.read_csv( '../input/covid19-global-forecasting-week-4/submission.csv')
train['Country_Region'].value_counts()[0:20].plot(kind = 'bar', grid = True)
import datetime as datetime

train['Dates'] = pd.to_datetime(train.Date)

train['Year'] = train.Dates.dt.year

train['Month'] = train.Dates.dt.month

train['Week'] = train.Dates.dt.week

train['Name of month'] = train.Dates.dt.month_name()

train['Week_day_name'] = train.Dates.dt.weekday_name

train['Day_of_year'] = train.Dates.dt.dayofyear
train.Fatalities.sum(), train.ConfirmedCases.sum()
x = train[['Year','Day_of_year', 'Month', 'Week']]

y = train['ConfirmedCases']
from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 4)
from sklearn.tree import DecisionTreeClassifier

dec = DecisionTreeClassifier(random_state=1)

dec.fit(x_train,y_train)
dec.score(x_test,y_test)
test.head()
import datetime as datetime

test['Dates'] = pd.to_datetime(test.Date)

test['Year'] = test.Dates.dt.year

test['Month'] = test.Dates.dt.month

test['Week'] = test.Dates.dt.week

test['Name of month'] = test.Dates.dt.month_name()

test['Week_day_name'] = test.Dates.dt.weekday_name

test['Day_of_year'] = test.Dates.dt.dayofyear
test.shape
x1 = test[['Year','Day_of_year', 'Month', 'Week']]
pred_1 = dec.predict(x1)
x2 = train[['Year','Day_of_year', 'Month', 'Week']]

y2 = train['Fatalities']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x2,y2, test_size = 0.2, random_state = 4)
from sklearn.tree import DecisionTreeClassifier

dec = DecisionTreeClassifier(random_state = 1)

dec.fit(x_train,y_train)

dec.score(x_test,y_test)
x3= test[['Year','Day_of_year', 'Month', 'Week']]
pred_2 = dec.predict(x3)
submission.head()
submission.ConfirmedCases = pred_1

submission.Fatalities = pred_2
submission.to_csv('submission.csv', index = False)