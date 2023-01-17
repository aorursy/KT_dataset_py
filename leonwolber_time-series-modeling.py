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
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.linear_model import LinearRegression
plt.style.use('fivethirtyeight')
pas = pd.read_csv('/kaggle/input/russian-passenger-air-service-20072020/russian_passenger_air_service_2.csv')

package = pd.read_csv('/kaggle/input/russian-passenger-air-service-20072020/russian_air_service_CARGO_AND_PARCELS.csv')
pas.head()
pas.isna().sum()/len(pas)*100
pas.head()
long = pas.melt(id_vars = ['Airport name', 'Year'],

         value_vars = ['January', 'February', 'March', 'April', 'May','June', 'July', 'August', 'September', 'October', 'November','December'])



long.rename(columns = {'variable':'month', 'value':'passengers'}, inplace = True)
# sort months in right order



months = ['January', 'February', 'March', 'April', 'May','June', 'July', 'August', 'September', 'October', 'November','December']

long['month'] = pd.Categorical(long['month'], categories=months, ordered=True)
long
per_year = long.groupby('Year')['passengers'].sum()/1000000

per_year = per_year.reset_index()
plt.figure(figsize=(20,11))



sns.barplot(data = per_year, x = 'Year', y = 'passengers')

plt.xlabel('Year', fontsize = 16)

plt.ylabel('Passengers in millions', fontsize = 16)

plt.title("Passengers per Year in Russia's airports", fontsize = 24)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)
ts =  long.groupby(['Year', 'month'])['passengers'].sum()
ts.to_frame()
ts = ts.reset_index()



time = range(1,169)

ts['time'] = time                            # create time variable

ts['pasmio'] = ts['passengers'] / 1000000    # change representation to millions
plt.figure(figsize = (20,10))



sns.scatterplot(data = ts, x = 'time', y = 'pasmio', color = 'black')

sns.lineplot(data = ts, x = 'time', y = 'pasmio', color = 'r')

plt.ylabel('Passengers in million', fontsize=  16)

plt.title("Time Series of Russias's airline passengers from 2007-2020", fontsize = 19)
ts
from statsmodels.tsa.stattools import adfuller

results = adfuller(ts['pasmio'])

print(results[1])
ts['log_pas'] = np.log(ts['pasmio'])
X = ts['time'].values

X = np.reshape(X, (len(X), 1))
y = ts['pasmio'].values
model = LinearRegression()

model.fit(X, y)
trend = model.predict(X)
plt.figure(figsize = (20,10))



sns.lineplot(data = trend)

sns.lineplot(data = ts, x = 'time', y = 'pasmio', color = 'r')

plt.ylabel('Passengers in million', fontsize=  16)

plt.title("Time Series of Russias's airline passengers from 2007-2020", fontsize = 19)
model
detrended = [y[i]-trend[i] for i in range(0, len(ts))]
plt.figure(figsize = (20,10))



plt.plot(detrended)

plt.show()
plt.figure(figsize = (20,10))



X = ts['pasmio'].values



diff = list()

for i in range(1, len(X)):

	value = X[i] - X[i - 1]

	diff.append(value)

plt.plot(diff)

plt.show()
from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_acf
detrended = pd.Series(detrended)
plot_acf(ts['pasmio'])
plot_acf(detrended)