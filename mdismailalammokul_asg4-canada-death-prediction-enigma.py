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
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
%matplotlib inline
import requests
import io
import random
sns.set(style="ticks", color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import timeit
from tensorflow.keras.callbacks import EarlyStopping
from fbprophet import Prophet
df = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv')
df.head()
df_canada = df.loc[df['Country/Region']=='Canada'].iloc[:,4:]
df_canada.shape
df_canada.isnull().sum()
death_daily = df_canada.sum(axis = 0)
death_daily.head(-15)
death_daily.index=pd.to_datetime(death_daily.index)
death_daily.head()
plt.figure(figsize=(9,7))
plt.plot(death_daily)
plt.title("Daily Death in Canada")
death_ca = death_daily.diff().fillna(death_daily[0]).astype(np.int)
death_ca.head(-5)
plt.figure(figsize=(9,7))
plt.plot(death_ca)
plt.title("Daily recovery")
death_ca=pd.DataFrame(death_ca)
death_ca.index.name = 'Datetime'
death_ca
col=['Daily_death']
death_ca.columns=col
death_ca
death_ca.shape
split_date = '2020-04-05'
train = death_ca.loc[death_ca.index <= split_date].copy()
test = death_ca.loc[death_ca.index > split_date].copy()
train.shape, test.shape
test
test \
    .rename(columns={'Daily_death': 'TEST SET'}) \
    .join(train.rename(columns={'Daily_death': 'TRAINING SET'}),
          how='outer') \
    .plot(figsize=(15,5), title='Death Daily', style='.')
plt.show()
# Format data for prophet model using ds and y
train.reset_index() \
    .rename(columns={'Datetime':'ds',
                     'Daily_death':'y'}).head()

# Setup and train model and fit
model = Prophet()
model.fit(train.reset_index() \
              .rename(columns={'Datetime':'ds',
                               'Daily_death':'y'}))
test_forcast = model.predict(df=test.reset_index() \
                                   .rename(columns={'Datetime':'ds'}))
test_forcast
# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = model.plot(test_forcast,ax=ax)
plt.show()
# Plot the components of the model
fig = model.plot_components(test_forcast)
# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(test.index, test['Daily_death'], color='r')
fig = model.plot(test_forcast, ax=ax)
