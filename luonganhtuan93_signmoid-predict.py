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
import numpy as np

import pandas as pd

import scipy.optimize as opt

%matplotlib inline

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_log_error
train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

train['Date'] = pd.to_datetime(train['Date'])

test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

test['Date'] = pd.to_datetime(test['Date'])

sub = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')

# sub = sub.set_index('ForecastId').reset_index(drop=True)

print(test.shape)

print(train.shape)

print(sub.shape)
train.head(5)
test.head(5)
sub
def sigmoid(t, M, beta, alpha):

    return M / (1 + np.exp(-beta * (t - alpha)))
NUMBER_UNCASEc= 47 

MAXIMUM = 1000

BOUNDS=(0, [2000, 1.0, 100])



x = list(range(len(train)))

min_date = min(train['Date'])

max_date = max(test['Date'])

cases = train['ConfirmedCases'].values

plt.plot(x, cases, label="Train Data")

print("Case:", cases)

popt, pcov = opt.curve_fit(sigmoid, x, cases, bounds=BOUNDS)

print(popt)

print(pcov)

M, beta, alpha = popt

plt.plot(x, sigmoid(x, M, beta, alpha), label="Predict")

# Place a legend to the right of this smaller subplot.

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)



plt.show()
M, beta, alpha = popt

x_forcast = list(range(train.shape[0], train.shape[0]+ test.shape[0]))

case_forecast = sigmoid(x_forcast, M, beta, alpha)

print("case_forecast",case_forecast)

sub["ConfirmedCases"] = [int(i) for i in case_forecast]

plt.plot(x_forcast, case_forecast)
deaths = train['Fatalities'].values

popt, pcov = opt.curve_fit(sigmoid, list(range(len(deaths))), deaths, bounds=BOUNDS)

M, beta, alpha = popt

death_forecast = sigmoid(x_forcast, M, beta, alpha)

print("death_forecast",death_forecast)

plt.plot(x_forcast, death_forecast)

sub["Fatalities"] = [int(i) for i in death_forecast]

sub.to_csv('submission.csv', index=False)
sub