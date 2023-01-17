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
import random
import seaborn as sns
sns.set(style="ticks", color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import timeit
from tensorflow.keras.callbacks import EarlyStopping
#Loading Data set
data_csv = pd.read_csv("../input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv")
#dataframe conversion
data_df=pd.DataFrame(data_csv)
#dataframe conversion
data_df=pd.DataFrame(data_csv)
df=data_df.iloc[:,4:]
df.head()
#checking the missing value
df.isnull().sum()
# daily total number of recovery
daily_recov = df.sum(axis=0)
daily_recov.head()
#changing the day time formal
daily_recov.index=pd.to_datetime(daily_recov.index)
daily_recov.head()
plt.figure(figsize=(9,7))
plt.plot(daily_recov)
plt.title("Cumulative daily recovery")
daily_recov=daily_recov.diff().fillna(daily_recov[0]).astype(np.int)
daily_recov.head()
plt.figure(figsize=(9,7))
plt.plot(daily_recov)
plt.title("Daily recovery")
daily_recov=pd.DataFrame(daily_recov)
daily_recov.index.name = 'Datetime'
daily_recov
col=['Recovery_Daily']
daily_recov.columns=col
split_date = '2020-04-05'
train = daily_recov.loc[daily_recov.index <= split_date].copy()
test = daily_recov.loc[daily_recov.index > split_date].copy()
test \
    .rename(columns={'Recovery_Daily': 'TEST SET'}) \
    .join(train.rename(columns={'Recovery_Daily': 'TRAINING SET'}),
          how='outer') \
    .plot(figsize=(15,5), title='Daily Recovery', style='.')
plt.show()
# Format data for prophet model using ds and y
train.reset_index() \
    .rename(columns={'Datetime':'ds',
                     'Recovery_Daily':'y'}).head()
from fbprophet import Prophet
# Setup and train model and fit
model = Prophet()
model.fit(train.reset_index() \
              .rename(columns={'Datetime':'ds',
                               'Recovery_Daily':'y'}))
# Predict on training set with model
test_fcst = model.predict(df=test.reset_index() \
                                   .rename(columns={'Datetime':'ds'}))
# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = model.plot(test_fcst,
                 ax=ax)
plt.show()
# Plot the components of the model
fig = model.plot_components(test_fcst)
# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(test.index, test['Recovery_Daily'], color='r')
fig = model.plot(test_fcst, ax=ax)
