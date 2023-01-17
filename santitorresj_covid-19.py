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
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

train.head(20)
test.head(20)
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import datetime

sns.set() # setting seaborn default for plots
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='inferno')
train.shape
test.shape
train.info()



test.info()
confirmed_country = train.groupby(['Country/Region']).agg({'ConfirmedCases':['sum']})

fatalities_country = train.groupby(['Country/Region']).agg({'Fatalities':['sum']})

confirmed_total_date = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date = train.groupby(['Date']).agg({'Fatalities':['sum']})

total_date = confirmed_total_date.join(fatalities_total_date)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

total_date.plot(ax=ax1)

ax1.set_title("Global confirmed cases")

ax1.set_ylabel("Number of cases")

fatalities_total_date.plot(ax=ax2, color='red')

ax2.set_title("Global deceased cases")

ax2.set_ylabel("Number of cases")
confirmed_country_Colombia = train[train['Country/Region']=='Colombia'].groupby(['Country/Region']).agg({'ConfirmedCases':['sum']})

fatalities_country_Colombia = train[train['Country/Region']=='Colombia'].groupby(['Country/Region']).agg({'Fatalities':['sum']})

confirmed_total_date_Colombia = train[train['Country/Region']=='Colombia'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Colombia = train[train['Country/Region']=='Colombia'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Colombia = confirmed_total_date_Colombia.join(fatalities_total_date_Colombia)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

total_date_Colombia.plot(ax=ax1)

ax1.set_title("Confirmed cases  in Colombia")

ax1.set_ylabel("Number of cases")

fatalities_total_date_Colombia.plot(ax=ax2, color='red')

ax2.set_title("Deceased cases in Colombia")

ax2.set_ylabel("Number of cases")
confirmed_country_Germany = train[train['Country/Region']=='Germany'].groupby(['Country/Region']).agg({'ConfirmedCases':['sum']})

fatalities_country_Germany = train[train['Country/Region']=='Germany'].groupby(['Country/Region']).agg({'Fatalities':['sum']})

confirmed_total_date_Germany = train[train['Country/Region']=='Germany'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Germany = train[train['Country/Region']=='Germany'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Germany = confirmed_total_date_Germany.join(fatalities_total_date_Germany)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

total_date_Germany.plot(ax=ax1)

ax1.set_title("Confirmed cases  in Germany")

ax1.set_ylabel("Number of cases")

fatalities_total_date_Germany.plot(ax=ax2, color='red')

ax2.set_title("Deceased cases in Germany")

ax2.set_ylabel("Number of cases")
confirmed_country_Italy = train[train['Country/Region']=='Italy'].groupby(['Country/Region']).agg({'ConfirmedCases':['sum']})

fatalities_country_Italy = train[train['Country/Region']=='Italy'].groupby(['Country/Region']).agg({'Fatalities':['sum']})

confirmed_total_date_Italy = train[train['Country/Region']=='Italy'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Italy = train[train['Country/Region']=='Italy'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Italy = confirmed_total_date_Italy.join(fatalities_total_date_Italy)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

total_date_Italy.plot(ax=ax1)

ax1.set_title("Confirmed cases  in Italy")

ax1.set_ylabel("Number of cases")

fatalities_total_date_Italy.plot(ax=ax2, color='red')

ax2.set_title("Deceased cases in Italy")

ax2.set_ylabel("Number of cases")
train.head()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import LabelEncoder
train['month'] = train['Date'].str.extract(r'[0-9]+[-]([0-9]+)[-]')

train['day'] = train['Date'].str.extract(r'[0-9]+[-][0-9]+[-]([0-9]+)')

train = train.drop('Date', axis = 1)



test['month'] = test['Date'].str.extract(r'[0-9]+[-]([0-9]+)[-]')

test['day'] = test['Date'].str.extract(r'[0-9]+[-][0-9]+[-]([0-9]+)')

test = test.drop('Date', axis = 1)



province = LabelEncoder()

train['Province/State'] = train['Province/State'].fillna('N')

test['Province/State'] = test['Province/State'].fillna('N')

province.fit(train['Province/State'].unique())

train['Province/State'] = province.transform(train['Province/State'])

test['Province/State'] = province.transform(test['Province/State'])



country = LabelEncoder()

country.fit(train['Country/Region'])

train['Country/Region'] = country.transform(train['Country/Region'])

test['Country/Region'] = country.transform(test['Country/Region'])



X = train.drop(['Id', 'ConfirmedCases', 'Fatalities'], axis = 1)

y1 = train.ConfirmedCases

y2 = train.Fatalities

test = test.drop('ForecastId', axis = 1)

x1_tr, x1_vld, y1_tr, y1_vld = train_test_split(X, y1, test_size = 0.2, random_state = 200)

x2_tr, x2_vld, y2_tr, y2_vld = train_test_split(X, y2, test_size = 0.2, random_state = 200)



tr1 = DecisionTreeRegressor()

tr2 = DecisionTreeRegressor()

tr1.fit(x1_tr, y1_tr)

tr2.fit(x2_tr, y2_tr)
from sklearn.metrics import mean_squared_log_error

tr1_pred = tr1.predict(x1_vld)

print('tr1 score : ', np.sqrt(mean_squared_log_error(y1_vld, tr1_pred)))

tr2_pred = tr2.predict(x2_vld)

print('tr2 score : ', np.sqrt(mean_squared_log_error(y2_vld, tr2_pred)))
confirmed_pred = tr1.predict(test)

fatalities_pred = tr2.predict(test)
submission.columns

submission['ConfirmedCases'] = confirmed_pred

submission['Fatalities'] = fatalities_pred

submission.to_csv('submission.csv', index=False)



submission