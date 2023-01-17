import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
ctba_temp = pd.read_csv("../input/temperature-timeseries-for-some-brazilian-cities/station_curitiba.csv")
ctba_temp.head(10)
ctba_temp.replace(999.90, np.nan, inplace=True)

ctba_temp.head(10)
df_ctba_temp = ctba_temp.set_index("YEAR", inplace=False)

df_ctba_temp
mean_temperature = ['metANN']

ctba_mean_temperature = df_ctba_temp[mean_temperature]
ctba_mean_temperature.head(10)
plt.figure(figsize=(12,6))

sns.lineplot(data = ctba_mean_temperature, legend=False)

plt.ylabel("Mean temperature (°C)")
ctba_temp_month = df_ctba_temp.loc[:, :"DEC"]
ctba_temp_2018 = ctba_temp_month[ctba_temp_month.index == 2018]

ctba_temp_2018
ctba_temp_2018 = ctba_temp_2018.T

ctba_temp_2018
plt.figure(figsize=(12, 6))

sns.lineplot(data=ctba_temp_2018, legend=False)

plt.ylabel("Temperature (°C)")

plt.title("Curitiba temperature mean in 2015")