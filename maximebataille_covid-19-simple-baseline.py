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
import csv

import pandas as pd

import seaborn as sns



from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
train.info()
test.info()
train.Id.unique().shape[0] == train.Id.shape[0]
test.ForecastId.unique().shape[0] == test.ForecastId.shape[0]
x_train = train[['Lat', 'Long', 'Date']]

y_train = train[['ConfirmedCases']]



x_test = test[['Lat', 'Long', 'Date']]
x_train.dtypes
def convert_date(date) :

    

    return ''.join(date.split('-'))
x_train['Date'] = x_train['Date'].apply(lambda x : convert_date(x))

x_test['Date'] = x_test['Date'].apply(lambda x : convert_date(x))
reg = RandomForestRegressor()

reg.fit(x_train, y_train)
predict_cases = reg.predict(x_test)
y_train = train['Fatalities']
reg = RandomForestRegressor()

reg.fit(x_train, y_train)
predict_fatalities = reg.predict(x_test)
submission['ConfirmedCases'] = predict_cases

submission['Fatalities'] = predict_fatalities
submission.to_csv('submission.csv', index = False)
submission.head()