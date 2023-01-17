import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



df = pd.read_csv('/home/anderson/Desktop/kaggledatasets/data.csv')
from fbprophet import Prophet
df.head()
df.drop('timestamp', axis = 1, inplace = True)
df.head()
df['date'] = pd.to_datetime(df['date'])
df.info()
df = pd.DataFrame(df.set_index('date').groupby(pd.TimeGrouper('1H')).sum())
df.head()
(df['number_people'] < 1).count()
#calcula o lucro da hora

df['Total'] = (df['number_people'])
df = df[['number_people']]
df.head()
df['ds'] = df.index

df.head()
df = df.rename(columns={'number_people': 'y'})
df['y'] = np.exp(df['y'])
from fbprophet import Prophet

m = Prophet(changepoint_prior_scale=0.001, mcmc_samples=500)

m.fit(df);
future = m.make_future_dataframe(periods=12000, freq='H')

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
import matplotlib.pyplot as plt

%matplotlib inline

m.plot(forecast);
m.plot_components(forecast);
import pandas as pd

import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

df = pd.read_csv('../input/data.csv')

## My code is on my github actually , as Kaggle does not accept some of the newest libraries I am using (Facebook Prophet)  :

#https://github.com/andersonamaral/my_machine_learning_studies/blob/master/Crowdness_At_Gym_With_Prophet_Forecast.ipynb
df.head()

df.drop('timestamp', axis = 1, inplace = True)
df['date'] = pd.to_datetime(df['date'])
df = pd.DataFrame(df.set_index('date').groupby(pd.TimeGrouper('1H')).sum())
df['Total'] = (df['number_people'])
df = df[['number_people']]

df.head()
df['ds'] = df.index

df.head()
df = df.rename(columns={'number_people': 'y'})

df['y'] = np.exp(df['y'])
from fbprophet import Prophet

m = Prophet(changepoint_prior_scale=0.001, mcmc_samples=500)

m.fit(df);
future = m.make_future_dataframe(periods=12, freq='M')

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
import matplotlib.pyplot as plt

%matplotlib inline

m.plot(forecast);