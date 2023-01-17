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

import matplotlib.pyplot as plt

%matplotlib inline

train = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

test = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')
X = train[['Id', 'Date', 'ConfirmedCases', 'Fatalities']]

Y = test[['ForecastId', 'Date']]

# peramalan untuk 12 maret sampai 23 april
import seaborn as sns

sns.pairplot(train)
X_train = X

X_train['Date'] = pd.to_datetime(X.Date , format = '%Y/%m/%d')

X_train.index = X_train.Date

X_train = X_train.drop(['Date', 'Id'], axis=1)
plt.plot(X_train['ConfirmedCases'], color='blue')

plt.plot(X_train['Fatalities'], color='red')

plt.show()
from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(X_train['ConfirmedCases'])

plot_pacf(X_train['ConfirmedCases']) #pacf dimulai dari 2
from statsmodels.tsa.ar_model import AR

X_cc = X_train['ConfirmedCases'].values

train_cc, test_cc = X_cc[1:len(X)-7], X_cc[len(X)-7:]

model = AR(train_cc)

model_fit = model.fit()

model_fit.summary()
predictions = model_fit.predict(start=len(train_cc), end=len(train_cc)+len(test_cc)-1, dynamic=False)

predictions
from sklearn.metrics import mean_squared_error

error = mean_squared_error(test_cc, predictions)

error
plt.plot(test_cc)

plt.plot(predictions, color='red')

plt.show()
X_cc = X_train['ConfirmedCases'].values

model = AR(X_cc)

model_fit = model.fit()

model_fit.summary()
predictions = model_fit.predict(start=len(X_cc), end=len(X_cc)+len(Y['Date'])-1)

predictions
test['ConfirmedCases'] = predictions
plt.plot(train['Date'], train['ConfirmedCases'], color='blue')

plt.plot(test['Date'], test['ConfirmedCases'], color='red')

plt.show()
plot_acf(X_train['Fatalities'])

plot_pacf(X_train['Fatalities']) #pacf dimulai dari 2
X_f = X_train['Fatalities'].values

train_f, test_f = X_f[1:len(X_f)-7], X_f[len(X_f)-7:]

model = AR(train_f)

model_fit = model.fit()

model_fit.summary()
predictions = model_fit.predict(start=len(train_f), end=len(train_f)+len(test_f)-1, dynamic=False)

predictions
error = mean_squared_error(test_f, predictions)

error
plt.plot(test_f)

plt.plot(predictions, color='red')

plt.show()
X_f = X_train['Fatalities'].values

model = AR(X_f)

model_fit = model.fit()

model_fit.summary()
predictions = model_fit.predict(start=len(X_f), end=len(X_f)+len(Y['Date'])-1)

predictions
test['Fatalities'] = predictions
plt.plot(train['Date'], train['Fatalities'], color='blue')

plt.plot(test['Date'], test['Fatalities'], color='red')

plt.show()
submission = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')

submission
submission = test[['ForecastId', 'ConfirmedCases', 'Fatalities']]