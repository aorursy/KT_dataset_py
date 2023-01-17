import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
ES = pd.read_excel('../input/airsense_compare/ESP_00983246_2018-08-01_2018-08-21.xlsx')
ES.head()
xls = pd.ExcelFile('../input/compare_device/airvisual162.xlsx')
li = []

for sheet_name in xls.sheet_names:

    df = pd.read_excel('../input/compare_device/airvisual162.xlsx', sheetname=sheet_name)

    li.append(df)
data_mt = pd.concat(li, axis=0, ignore_index=True).drop_duplicates()
data_mt = pd.concat(li, axis=0).drop_duplicates().dropna()
data_mt.head()
data_mt = data_mt.drop(columns=['Date','Time'])
data_mt['Datetime'] =  pd.to_datetime(data_mt['Datetime'], format='%m/%d/%y %H:%M:%S')
data_mt.head()
data_mt = data_mt.groupby(pd.Grouper(key='Datetime', freq='5T')).mean().dropna()
data_mt.head()
ES = ES.groupby(pd.Grouper(key='Time Stamp', freq='5T')).mean().dropna()
ES.head()
data_mt = data_mt.rename(index=str, columns={"PM2_5(ug/m3)": "PM2p5", "Temperature(C)": "Temperature", "Humidity(%RH)" : 'Humidity',"PM10(ug/m3)": 'PM10'})
ES = ES.rename(index=str, columns={" Temperature": "Temperature", " Humidity" : 'Humidity'})
ES.head()
data_mt.head()
ES.shape
data_mt.shape
df_inner = pd.merge(ES, data_mt, right_index=True, left_index=True)
df_inner.head()
df_inner.shape
print('PM2.5: %s PM10: %s Temperature: %s Humidity: %s' %(df_inner.PM2p5_x.mean(), df_inner.PM10_x.mean(), df_inner.Temperature_x.mean(), df_inner.Humidity_x.mean()))
print('PM2.5: %s PM10: %s Temperature: %s Humidity: %s' %(df_inner.PM2p5_y.mean(),  df_inner.PM10_y.mean(), df_inner.Temperature_y.mean(), df_inner.Humidity_y.mean()))
from scipy.stats import ttest_ind
ttest_ind(df_inner.Humidity_x, df_inner.Humidity_y)
ttest_ind(df_inner.Temperature_x, df_inner.Temperature_y)
ttest_ind(df_inner.PM10_x, df_inner.PM10_y)
ttest_ind(df_inner.PM2p5_x, df_inner.PM2p5_y)
import matplotlib.pyplot as plt
df_inner.Humidity_x.plot(label="ESP_00983246", figsize=(20,10))

df_inner.Humidity_y.plot(label='AirVisual 162')

plt.legend()

plt.title("Humidity")

plt.show()


df_inner.Temperature_x.plot(label="ESP_00983246", figsize=(20,10))

df_inner.Temperature_y.plot(label='AirVisual 162')

plt.legend()

plt.title("Temperature")

plt.show()
df_inner.PM10_x.plot(label="ESP_00983246", figsize=(20,10))

df_inner.PM10_y.plot(label='AirVisual 162')

plt.legend()

plt.title("PM10")

plt.show()
df_inner.PM2p5_x.plot(label="ESP_00983246", figsize=(20,10))

df_inner.PM2p5_y.plot(label='AirVisual 162')

plt.legend()

plt.title("PM2p5")

plt.show()
from scipy.stats.stats import pearsonr   
pearsonr(df_inner.Humidity_x,df_inner.Humidity_y)
pearsonr(df_inner.Temperature_x,df_inner.Temperature_y)
pearsonr(df_inner.PM10_x,df_inner.PM10_y)
pearsonr(df_inner.PM2p5_x,df_inner.PM2p5_y)