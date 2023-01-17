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

import numpy as np
tscovid = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv', sep=',')
artime = tscovid.iloc[0:405, 4:54]
days = pd.date_range('2020/1/22', periods=50, freq='D')
artime = artime.T
artime.head()
artime = pd.DataFrame(artime)
df = tscovid.iloc[0:405, 4:54]
total = df.sum(axis=0)
total
artime['Total'] = total
artime['Day'] = days
artime.set_index('Day', inplace=True)
artime['Total']
from statsmodels.tsa.ar_model import AR,ARResults
train = artime.iloc[:42]

test = artime.iloc[42:]
import warnings

warnings.filterwarnings("ignore")
model = AR(train['Total'])

AR1fit = model.fit(maxlag=2,method='cmle')

print(f'Lag: {AR1fit.k_ar}')

print(f'Coefficients:\n{AR1fit.params}')
start=len(train)

end=len(train)+len(test)-1

predictions1 = AR1fit.predict(start=start, end=end).rename('AR(1) Predictions')
for i in range(len(predictions1)):

    print(f"predicted={predictions1[i]}, expected={test['Total'][i]}")
test['Total'].plot(legend=True)

predictions1.plot(legend=True,figsize=(12,6));
AR2fit = model.fit(maxlag=2,method='cmle')

print(f'Lag: {AR2fit.k_ar}')

print(f'Coefficients:\n{AR2fit.params}')
start=len(train)

end=len(train)+len(test)-1

predictions2 = AR2fit.predict(start=start, end=end).rename('AR(2) Predictions')
test['Total'].plot(legend=True)

predictions1.plot(legend=True)

predictions2.plot(legend=True,figsize=(12,6));
ARfit = model.fit(maxlag=2,method='cmle')

print(f'Lag: {ARfit.k_ar}')

print(f'Coefficients:\n{ARfit.params}')
start = len(train)

end = len(train)+len(test)-1

rename = f'AR(12) Predictions'



predictions11 = ARfit.predict(start=start,end=end).rename(rename)
test['Total'].plot(legend=True)

predictions1.plot(legend=True)

predictions2.plot(legend=True)

predictions11.plot(legend=True,figsize=(12,6));
from sklearn.metrics import mean_squared_error



labels = ['AR(1)','AR(2)','AR(11)']

preds = [predictions1, predictions2, predictions11]  # these are variables, not strings!



for i in range(3):

    error = mean_squared_error(test['Total'], preds[i])

    print(f'{labels[i]} Error: {error:11.10}')
model = AR(artime['Total'])



# Next, fit the model

ARfit = model.fit(maxlag=8)



# Make predictions

fcast = ARfit.predict(start=len(artime), end=len(artime)+20).rename('Forecast')



# Plot the results

artime['Total'].plot(legend=True)

fcast.plot(legend=True, grid=True, figsize=(12,6));
print('Expectative for coronavirus deaths till April 01 is', fcast)