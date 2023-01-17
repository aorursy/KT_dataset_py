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

import warnings

warnings.filterwarnings("ignore")
import matplotlib

import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16, 10)

import plotly_express as px
from pathlib import Path



base_path = Path('/kaggle/input/air-quality-madrid/csvs_per_year/csvs_per_year/')



files = base_path.rglob('*.csv')
year_dfs = []

for file in files:

    df = pd.read_csv(file)

    print(df.shape)

    year_dfs.append(df)

    

raw_data = pd.concat(year_dfs, axis=0)

raw_data = raw_data[['date', 'station', 'O_3', 'CO', 'SO_2']]

raw_data['date'] = pd.to_datetime(raw_data['date'], format='%Y-%m-%d %H:%M:%S')

agg_data = raw_data[['date', 'O_3', 'CO', 'SO_2']].groupby('date').mean()

agg_data = agg_data.sort_index()
agg_data = agg_data.reset_index()

agg_data.head()
data = agg_data.groupby(agg_data.date.dt.date).mean()
data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
plt.plot(data['O_3'])

plt.plot(data['SO_2'])

plt.plot(data['CO'])

plt.legend()
data['O_3'].plot()
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(data['O_3'], model='additive')

fig = decompose_result.plot()
from fbprophet import Prophet

model = Prophet().fit(pd.DataFrame({'ds': data.index, 'y': data.O_3}))
forecast_dates = model.make_future_dataframe(365 * 2)

forecast = model.predict(forecast_dates)
fig = plt.figure(figsize=(16, 10))

fig = model.plot(forecast)
fig2 = model.plot_components(forecast)
forecast