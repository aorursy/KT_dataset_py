import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
forecast_moment = pd.datetime(2011, 9, 2, 0, 0, 0)

forecast_horizon = 24

column_name = 'consumption_eur'         # column to forecast

time_index_name = 'timestep'

M = 144                                 # model on the most similar pattern parameter
data_source = pd.read_csv('/kaggle/input/russian-wholesale-electricity-market/RU_Electricity_Market_PZ_dayahead_price_volume.csv')

data_source.index = pd.to_datetime(data_source[time_index_name], format='%Y-%m-%d %H:%M')

data_source.index.name = time_index_name

data = data_source[column_name]

data.head()
pattern_latest_available_range = pd.date_range(forecast_moment - pd.Timedelta(M, unit='H'),

                                               forecast_moment - pd.Timedelta(1, unit='H'), freq='H')

pattern_latest_available = data.loc[pattern_latest_available_range]
looping_dates_range = pd.date_range(data.index[0],

                                    forecast_moment - pd.Timedelta(M + forecast_horizon, unit='H'), freq='D')

similarity_measure = []

time_delay = []



# Looping through timeseries history



for d in looping_dates_range:



    pattern_temp_range = pd.date_range(d, d + pd.Timedelta(M-1, unit='H'), freq='H')

    ds = data.loc[pattern_temp_range].values

    time_delay.append(d)



    if np.sum(ds) == 0:

        similarity_measure.append(0)

    else:

        # Similarity measure = abs of linear correlation

        c = np.abs(np.corrcoef(ds, pattern_latest_available.values))

        similarity_measure.append(c[0, 1])
similarity = pd.DataFrame(similarity_measure, index=time_delay, columns=['similarity'])

max_similarity = np.max(similarity.values)

max_time_delay = similarity[similarity.values == max_similarity].index

max_similarity_pattern_range = pd.date_range(max_time_delay[0],

                                             max_time_delay[0] + pd.Timedelta(M-1, unit='H'), freq='H')

max_similarity_pattern = data.loc[max_similarity_pattern_range]
regress = LinearRegression()

x = np.column_stack((max_similarity_pattern.values, np.ones(len(max_similarity_pattern))))

regress.fit(x, pattern_latest_available.values)

max_similarity_pattern_model = regress.predict(x)
base_pattern_range = pd.date_range(max_similarity_pattern.index[-1] + pd.Timedelta(1, unit='H'),

                                   max_similarity_pattern.index[-1] + pd.Timedelta(forecast_horizon, unit='H'), freq='H')

base_pattern = data.loc[base_pattern_range]
forecast_range = pd.date_range(pattern_latest_available.index[-1] + pd.Timedelta(1, unit='H'),

                               pattern_latest_available.index[-1] + pd.Timedelta(forecast_horizon, unit='H'), freq='H')

x = np.column_stack((base_pattern.values, np.ones(len(base_pattern))))

y = regress.predict(x)

forecast = pd.DataFrame(y, index=forecast_range, columns=[column_name])
actual = data.loc[forecast_range]

mae = np.mean(np.abs(actual.values.ravel() - forecast.values.ravel()))

mape = np.mean(np.abs((actual.values.ravel() - forecast.values.ravel()) / actual.values.ravel())) * 100

error_line = 'MAE = %2.2f MWh, MAPE = %2.2f %% ' % (mae, mape)
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 20))

ax1.plot(actual.index, actual.values, label='Actuals')

ax1.plot(forecast.index, forecast.values, label='Forecast')

ax1.set_title('Forecast for ' + column_name + ': ' + error_line, fontsize=20)

ax1.legend()



ax2.plot(max_similarity_pattern.index, pattern_latest_available.values, label='Latest available pattern (x-shifted)')

ax2.plot(max_similarity_pattern.index, max_similarity_pattern.values, label='Max similarity pattern')

ax2.plot(max_similarity_pattern.index, max_similarity_pattern_model, label='Model of max similarity pattern')

ax2.plot(base_pattern.index, base_pattern.values, label='Base pattern values')

ax2.plot(base_pattern.index, forecast.values, label='Forecast (x-shifted)')

ax2.legend()



plt.subplots_adjust(hspace=0.3)

plt.show()