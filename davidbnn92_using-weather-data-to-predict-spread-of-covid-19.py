import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error as MAE



for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)





train = pd.read_csv('/kaggle/input/weather-data-for-covid19-data-analysis/training_data_with_weather_info_week_2.csv')

#train.drop('Unnamed: 0', axis=1, inplace=True)

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

display(train.head())

display(test.head())
df = train[train['country+province']=='Italy-']

df = df[df['ConfirmedCases']>0]

sns.lineplot(data=df.set_index('day_from_jan_first')['ConfirmedCases'])
sns.lineplot(data=df.set_index('day_from_jan_first')[['temp', 'min', 'max']])
col = 'ConfirmedCases'



s = df[col].shift(periods=1).fillna(0)

df[col + '_diff'] = df[col]-s

df[col + '_new_percent'] = (df[col] / s) - 1

df[col + '_new_percent'].replace([np.inf, -np.inf], 1, inplace=True)



sns.lineplot(data=df.set_index('day_from_jan_first')[['ConfirmedCases_diff']])
sns.lineplot(data=df.set_index('day_from_jan_first')[['ConfirmedCases_new_percent']])
corrmat = df[['ConfirmedCases', 'Fatalities', 

       'temp', 'min', 'max', 'stp', 'wdsp', 'prcp', 'fog', 'ConfirmedCases_diff',

       'ConfirmedCases_new_percent']].corr()

plt.subplots(figsize=(25,25))

sns.heatmap(corrmat, vmax=0.9, square=True, annot=True)
df['t'] = df.day_from_jan_first - df.day_from_jan_first.min()

df['t^2'] = df['t']**2



X = df[df['t'] < (df['t'].max() - 7)]

X_t = df[df['t'] >= (df['t'].max() - 7)]

y = X.ConfirmedCases_diff.apply(np.log).replace([np.inf, -np.inf], 0)

y_t = X_t.ConfirmedCases_diff.apply(np.log).replace([np.inf, -np.inf], 0)

cols = ['temp', 'min', 'max', 'stp', 'wdsp', 'prcp',

       'fog', 't', 't^2']



print('Target mean is : {}'.format(y.mean()))

print('Target std is : {}'.format(y.std()))



regressor = DecisionTreeRegressor()

regressor.fit(X[cols], y)

pred = regressor.predict(X_t[cols])

print('MAE: {}'.format(MAE(y_t, pred)))



sns.lineplot(data=pd.DataFrame({'validation target': y_t, 'predictions': pred}))