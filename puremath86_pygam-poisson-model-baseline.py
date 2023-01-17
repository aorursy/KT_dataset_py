# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

%matplotlib inline
# Retriving Dataset

df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df_confirmed.head()
df_deaths.head()
df_confirmed = df_confirmed.rename(columns={"Province/State":"state","Country/Region": "country"})

df_deaths = df_deaths.rename(columns={"Province/State":"state","Country/Region": "country"})
df_confirmed.head()
df_deaths.head()
df_deaths[df_deaths.country == 'US']
df_confirmed[df_confirmed.country == 'US']
tmp = df_deaths[df_deaths.country == 'US'].T

values = tmp.tail(116-4).values

dates = tmp.tail(116-4).index.tolist()



df_usa = pd.DataFrame(values, columns=['Deaths'], index=dates)

df_usa.tail()
tmp = df_confirmed[df_confirmed.country == 'US'].T

values = tmp.tail(116-4).values



df_usa['Cases'] = values

df_usa.tail()
SIZE = (20,8)



plt.figure(figsize=SIZE)

df_usa.Deaths.plot()

plt.show()
plt.figure(figsize=SIZE)

df_usa.Cases.plot()

plt.show()
df_usa = df_usa.reset_index().rename({'index':'Date'}, axis=1).reset_index().rename({'index':'t'}, axis=1)

df_usa.index = df_usa.Date

del df_usa['Date']



df_usa.tail()
df_usa['Deaths'] = df_usa['Deaths'].diff()

df_usa['Cases'] = df_usa['Cases'].diff()

df_usa = df_usa.dropna()

dates = df_usa.index
plt.figure(figsize=SIZE)

df_usa.Deaths.plot()

plt.show()
plt.figure(figsize=SIZE)

df_usa.Cases.plot()

plt.show()
corrs = []

for i in range(20):

    Y = np.stack((df_usa.Cases.shift(i).fillna(0).values, df_usa.Deaths.values), axis=0).astype(np.float32)

    corr = np.corrcoef(Y)[1,0]

    corrs.append(corr)

    

corrs
df_usa['Cases'] = df_usa.Cases.shift(5).fillna(0)
used_cols = ['t', 'Cases']

target = 'Deaths'



X, y = df_usa[used_cols].values, df_usa[target].values
from scipy.signal import savgol_filter



smooth = savgol_filter(y, 11, 2)

smooth = savgol_filter(smooth, 11, 1)

smooth = savgol_filter(smooth, 5, 1)
!pip install pygam
from pygam import PoissonGAM



gam = PoissonGAM().gridsearch(X, smooth.round())
gam.summary()
X_new = X[:90]+112



y_hat = gam.predict(X)

y_pred = gam.predict(X_new)

lb, ub = gam.confidence_intervals(X_new, width=0.997).T



plt.figure(figsize=SIZE)

plt.plot(X[:,0], y)

plt.plot(X[:,0], y_hat)

plt.plot(X[:,0], smooth)

plt.plot(X_new[:,0], y_pred)

plt.plot(X_new[:,0], lb)

plt.plot(X_new[:,0], ub)

plt.show()
y0 = y.cumsum().max()



plt.figure(figsize=SIZE)

plt.plot(X[:,0], y.cumsum())

plt.plot(X[:,0], y_hat.cumsum())

plt.plot(X[:,0], smooth.cumsum())

plt.plot(X_new[:,0], y0+y_pred.cumsum())

plt.plot(X_new[:,0], y0+lb.cumsum())

plt.plot(X_new[:,0], y0+ub.cumsum())

plt.show()