# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
dat1 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv", parse_dates=['DATE_TIME'], infer_datetime_format=True)
dat2 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv", parse_dates=['DATE_TIME'], infer_datetime_format=True)
dat3 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv", parse_dates=['DATE_TIME'], infer_datetime_format=True)
dat4 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv", parse_dates=['DATE_TIME'], infer_datetime_format=True)
print(dat1.head())
print(dat2.head())
print(dat3.head())
print(dat4.head())
dat_plant_power_1_agg_mst = dat1.groupby('DATE_TIME').agg({
                                                    'DC_POWER': 'mean',
                                                    'AC_POWER': 'mean',
                                                    'DAILY_YIELD': 'mean',
                                                    'TOTAL_YIELD': 'mean'
                                                }).reset_index()

dat_plant_1_temp1 = dat_plant_power_1_agg_mst.copy()
dat_plant_1_temp1 = dat_plant_1_temp1.drop(['TOTAL_YIELD'], axis=1)
dat_plant_1_temp2 = dat_plant_power_1_agg_mst.copy()
dat_plant_1_temp2 = dat_plant_1_temp2.drop(['TOTAL_YIELD'], axis=1)
dat_plant_1_temp3 = dat_plant_power_1_agg_mst.copy()
dat_plant_1_temp3 = dat_plant_1_temp3.drop(['TOTAL_YIELD'], axis=1)
dat_plant_power_1_agg_mst = dat_plant_power_1_agg_mst.drop(['DC_POWER','AC_POWER','DAILY_YIELD'], axis=1)

dat_plant_1_temp1["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_1_agg_mst["DATE_TIME"]) + pd.DateOffset(1)
dat_plant_1_temp2["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_1_agg_mst["DATE_TIME"]) + pd.DateOffset(2)
dat_plant_1_temp3["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_1_agg_mst["DATE_TIME"]) + pd.DateOffset(3)

dat_plant_power_1_agg_mst = pd.merge(dat_plant_power_1_agg_mst, dat_plant_1_temp1, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_1_agg_mst = pd.merge(dat_plant_power_1_agg_mst, dat_plant_1_temp2, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_1_agg_mst = pd.merge(dat_plant_power_1_agg_mst, dat_plant_1_temp3, left_on='DATE_TIME', right_on='DATE_TIME')

dat_plant_power_1_agg_mst["HOUR"] = dat_plant_power_1_agg_mst['DATE_TIME'].dt.hour
dat_plant_power_1_agg_mst["MINUTE"] = dat_plant_power_1_agg_mst['DATE_TIME'].dt.minute
dat_plant_weather_1_agg_mst = dat2.groupby('DATE_TIME').agg({
                                                    'AMBIENT_TEMPERATURE': 'mean',
                                                    'MODULE_TEMPERATURE': 'mean',
                                                    'IRRADIATION': 'mean'
                                                }).reset_index()

dat_plant_1_weather_agg_temp1 = dat_plant_weather_1_agg_mst.copy()
dat_plant_1_weather_agg_temp2 = dat_plant_weather_1_agg_mst.copy()
dat_plant_1_weather_agg_temp3 = dat_plant_weather_1_agg_mst.copy()

dat_plant_1_weather_agg_temp1["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_1_agg_mst["DATE_TIME"]) + pd.DateOffset(1)
dat_plant_1_weather_agg_temp2["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_1_agg_mst["DATE_TIME"]) + pd.DateOffset(2)
dat_plant_1_weather_agg_temp3["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_1_agg_mst["DATE_TIME"]) + pd.DateOffset(3)

dat_plant_weather_1_agg_mst = pd.merge(dat_plant_weather_1_agg_mst, dat_plant_1_weather_agg_temp1, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_1_agg_mst = pd.merge(dat_plant_weather_1_agg_mst, dat_plant_1_weather_agg_temp2, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_1_agg_mst = pd.merge(dat_plant_weather_1_agg_mst, dat_plant_1_weather_agg_temp3, left_on='DATE_TIME', right_on='DATE_TIME')

dat_plant_1_mst = pd.merge(dat_plant_power_1_agg_mst, dat_plant_weather_1_agg_mst, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_2_agg_mst = dat3.groupby('DATE_TIME').agg({
                                                    'DC_POWER': 'mean',
                                                    'AC_POWER': 'mean',
                                                    'DAILY_YIELD': 'mean',
                                                    'TOTAL_YIELD': 'mean'
                                                }).reset_index()

dat_plant_2_temp1 = dat_plant_power_2_agg_mst.copy()
dat_plant_2_temp1 = dat_plant_2_temp1.drop(['TOTAL_YIELD'], axis=1)
dat_plant_2_temp2 = dat_plant_power_2_agg_mst.copy()
dat_plant_2_temp2 = dat_plant_2_temp2.drop(['TOTAL_YIELD'], axis=1)
dat_plant_2_temp3 = dat_plant_power_2_agg_mst.copy()
dat_plant_2_temp3 = dat_plant_2_temp3.drop(['TOTAL_YIELD'], axis=1)
dat_plant_power_2_agg_mst = dat_plant_power_2_agg_mst.drop(['DC_POWER','AC_POWER','DAILY_YIELD'], axis=1)

dat_plant_2_temp1["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_2_agg_mst["DATE_TIME"]) + pd.DateOffset(1)
dat_plant_2_temp2["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_2_agg_mst["DATE_TIME"]) + pd.DateOffset(2)
dat_plant_2_temp3["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_2_agg_mst["DATE_TIME"]) + pd.DateOffset(3)

dat_plant_power_2_agg_mst = pd.merge(dat_plant_power_2_agg_mst, dat_plant_2_temp1, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_2_agg_mst = pd.merge(dat_plant_power_2_agg_mst, dat_plant_2_temp2, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_2_agg_mst = pd.merge(dat_plant_power_2_agg_mst, dat_plant_2_temp3, left_on='DATE_TIME', right_on='DATE_TIME')

dat_plant_power_2_agg_mst["HOUR"] = dat_plant_power_2_agg_mst['DATE_TIME'].dt.hour
dat_plant_power_2_agg_mst["MINUTE"] = dat_plant_power_2_agg_mst['DATE_TIME'].dt.minute
dat_plant_weather_2_agg_mst = dat4.groupby('DATE_TIME').agg({
                                                    'AMBIENT_TEMPERATURE': 'mean',
                                                    'MODULE_TEMPERATURE': 'mean',
                                                    'IRRADIATION': 'mean'
                                                }).reset_index()

dat_plant_2_weather_agg_temp1 = dat_plant_weather_2_agg_mst.copy()
dat_plant_2_weather_agg_temp2 = dat_plant_weather_2_agg_mst.copy()
dat_plant_2_weather_agg_temp3 = dat_plant_weather_2_agg_mst.copy()

dat_plant_2_weather_agg_temp1["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_2_agg_mst["DATE_TIME"]) + pd.DateOffset(1)
dat_plant_2_weather_agg_temp2["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_2_agg_mst["DATE_TIME"]) + pd.DateOffset(2)
dat_plant_2_weather_agg_temp3["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_2_agg_mst["DATE_TIME"]) + pd.DateOffset(3)

dat_plant_weather_2_agg_mst = pd.merge(dat_plant_weather_2_agg_mst, dat_plant_2_weather_agg_temp1, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_2_agg_mst = pd.merge(dat_plant_weather_2_agg_mst, dat_plant_2_weather_agg_temp2, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_2_agg_mst = pd.merge(dat_plant_weather_2_agg_mst, dat_plant_2_weather_agg_temp3, left_on='DATE_TIME', right_on='DATE_TIME')

dat_plant_2_mst = pd.merge(dat_plant_power_2_agg_mst, dat_plant_weather_2_agg_mst, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_1_mst.head()
dat_plant_2_mst.head()
dat_3d = pd.concat([dat_plant_1_mst, dat_plant_2_mst])
dat_plant_power_1_agg_mst = dat1.groupby('DATE_TIME').agg({
                                                    'DC_POWER': 'mean',
                                                    'AC_POWER': 'mean',
                                                    'DAILY_YIELD': 'mean',
                                                    'TOTAL_YIELD': 'mean'
                                                }).reset_index()

dat_plant_1_temp1 = dat_plant_power_1_agg_mst.copy()
dat_plant_1_temp1 = dat_plant_1_temp1.drop(['TOTAL_YIELD'], axis=1)
dat_plant_1_temp2 = dat_plant_power_1_agg_mst.copy()
dat_plant_1_temp2 = dat_plant_1_temp2.drop(['TOTAL_YIELD'], axis=1)
dat_plant_1_temp3 = dat_plant_power_1_agg_mst.copy()
dat_plant_1_temp3 = dat_plant_1_temp3.drop(['TOTAL_YIELD'], axis=1)
dat_plant_1_temp4 = dat_plant_power_1_agg_mst.copy()
dat_plant_1_temp4 = dat_plant_1_temp4.drop(['TOTAL_YIELD'], axis=1)
dat_plant_1_temp5 = dat_plant_power_1_agg_mst.copy()
dat_plant_1_temp5 = dat_plant_1_temp5.drop(['TOTAL_YIELD'], axis=1)
dat_plant_1_temp6 = dat_plant_power_1_agg_mst.copy()
dat_plant_1_temp6 = dat_plant_1_temp6.drop(['TOTAL_YIELD'], axis=1)
dat_plant_1_temp7 = dat_plant_power_1_agg_mst.copy()
dat_plant_1_temp7 = dat_plant_1_temp7.drop(['TOTAL_YIELD'], axis=1)
dat_plant_power_1_agg_mst = dat_plant_power_1_agg_mst.drop(['DC_POWER','AC_POWER','DAILY_YIELD'], axis=1)

dat_plant_1_temp1["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_1_agg_mst["DATE_TIME"]) + pd.DateOffset(1)
dat_plant_1_temp2["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_1_agg_mst["DATE_TIME"]) + pd.DateOffset(2)
dat_plant_1_temp3["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_1_agg_mst["DATE_TIME"]) + pd.DateOffset(3)
dat_plant_1_temp4["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_1_agg_mst["DATE_TIME"]) + pd.DateOffset(4)
dat_plant_1_temp5["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_1_agg_mst["DATE_TIME"]) + pd.DateOffset(5)
dat_plant_1_temp6["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_1_agg_mst["DATE_TIME"]) + pd.DateOffset(6)
dat_plant_1_temp7["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_1_agg_mst["DATE_TIME"]) + pd.DateOffset(7)

dat_plant_power_1_agg_mst = pd.merge(dat_plant_power_1_agg_mst, dat_plant_1_temp1, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_1_agg_mst = pd.merge(dat_plant_power_1_agg_mst, dat_plant_1_temp2, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_1_agg_mst = pd.merge(dat_plant_power_1_agg_mst, dat_plant_1_temp3, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_1_agg_mst = pd.merge(dat_plant_power_1_agg_mst, dat_plant_1_temp4, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_1_agg_mst = pd.merge(dat_plant_power_1_agg_mst, dat_plant_1_temp5, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_1_agg_mst = pd.merge(dat_plant_power_1_agg_mst, dat_plant_1_temp6, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_1_agg_mst = pd.merge(dat_plant_power_1_agg_mst, dat_plant_1_temp7, left_on='DATE_TIME', right_on='DATE_TIME')

dat_plant_power_1_agg_mst["HOUR"] = dat_plant_power_1_agg_mst['DATE_TIME'].dt.hour
dat_plant_power_1_agg_mst["MINUTE"] = dat_plant_power_1_agg_mst['DATE_TIME'].dt.minute
dat_plant_weather_1_agg_mst = dat2.groupby('DATE_TIME').agg({
                                                    'AMBIENT_TEMPERATURE': 'mean',
                                                    'MODULE_TEMPERATURE': 'mean',
                                                    'IRRADIATION': 'mean'
                                                }).reset_index()

dat_plant_1_weather_agg_temp1 = dat_plant_weather_1_agg_mst.copy()
dat_plant_1_weather_agg_temp2 = dat_plant_weather_1_agg_mst.copy()
dat_plant_1_weather_agg_temp3 = dat_plant_weather_1_agg_mst.copy()
dat_plant_1_weather_agg_temp4 = dat_plant_weather_1_agg_mst.copy()
dat_plant_1_weather_agg_temp5 = dat_plant_weather_1_agg_mst.copy()
dat_plant_1_weather_agg_temp6 = dat_plant_weather_1_agg_mst.copy()
dat_plant_1_weather_agg_temp7 = dat_plant_weather_1_agg_mst.copy()

dat_plant_1_weather_agg_temp1["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_1_agg_mst["DATE_TIME"]) + pd.DateOffset(1)
dat_plant_1_weather_agg_temp2["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_1_agg_mst["DATE_TIME"]) + pd.DateOffset(2)
dat_plant_1_weather_agg_temp3["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_1_agg_mst["DATE_TIME"]) + pd.DateOffset(3)
dat_plant_1_weather_agg_temp4["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_1_agg_mst["DATE_TIME"]) + pd.DateOffset(4)
dat_plant_1_weather_agg_temp5["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_1_agg_mst["DATE_TIME"]) + pd.DateOffset(5)
dat_plant_1_weather_agg_temp6["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_1_agg_mst["DATE_TIME"]) + pd.DateOffset(6)
dat_plant_1_weather_agg_temp7["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_1_agg_mst["DATE_TIME"]) + pd.DateOffset(7)

dat_plant_weather_1_agg_mst = pd.merge(dat_plant_weather_1_agg_mst, dat_plant_1_weather_agg_temp1, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_1_agg_mst = pd.merge(dat_plant_weather_1_agg_mst, dat_plant_1_weather_agg_temp2, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_1_agg_mst = pd.merge(dat_plant_weather_1_agg_mst, dat_plant_1_weather_agg_temp3, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_1_agg_mst = pd.merge(dat_plant_weather_1_agg_mst, dat_plant_1_weather_agg_temp4, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_1_agg_mst = pd.merge(dat_plant_weather_1_agg_mst, dat_plant_1_weather_agg_temp5, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_1_agg_mst = pd.merge(dat_plant_weather_1_agg_mst, dat_plant_1_weather_agg_temp6, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_1_agg_mst = pd.merge(dat_plant_weather_1_agg_mst, dat_plant_1_weather_agg_temp7, left_on='DATE_TIME', right_on='DATE_TIME')

dat_plant_1_mst = pd.merge(dat_plant_power_1_agg_mst, dat_plant_weather_1_agg_mst, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_2_agg_mst = dat3.groupby('DATE_TIME').agg({
                                                    'DC_POWER': 'mean',
                                                    'AC_POWER': 'mean',
                                                    'DAILY_YIELD': 'mean',
                                                    'TOTAL_YIELD': 'mean'
                                                }).reset_index()

dat_plant_2_temp1 = dat_plant_power_2_agg_mst.copy()
dat_plant_2_temp1 = dat_plant_2_temp1.drop(['TOTAL_YIELD'], axis=1)
dat_plant_2_temp2 = dat_plant_power_2_agg_mst.copy()
dat_plant_2_temp2 = dat_plant_2_temp2.drop(['TOTAL_YIELD'], axis=1)
dat_plant_2_temp3 = dat_plant_power_2_agg_mst.copy()
dat_plant_2_temp3 = dat_plant_2_temp3.drop(['TOTAL_YIELD'], axis=1)
dat_plant_2_temp4 = dat_plant_power_2_agg_mst.copy()
dat_plant_2_temp4 = dat_plant_2_temp4.drop(['TOTAL_YIELD'], axis=1)
dat_plant_2_temp5 = dat_plant_power_2_agg_mst.copy()
dat_plant_2_temp5 = dat_plant_2_temp5.drop(['TOTAL_YIELD'], axis=1)
dat_plant_2_temp6 = dat_plant_power_2_agg_mst.copy()
dat_plant_2_temp6 = dat_plant_2_temp6.drop(['TOTAL_YIELD'], axis=1)
dat_plant_2_temp7 = dat_plant_power_2_agg_mst.copy()
dat_plant_2_temp7 = dat_plant_2_temp7.drop(['TOTAL_YIELD'], axis=1)
dat_plant_power_2_agg_mst = dat_plant_power_2_agg_mst.drop(['DC_POWER','AC_POWER','DAILY_YIELD'], axis=1)

dat_plant_2_temp1["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_2_agg_mst["DATE_TIME"]) + pd.DateOffset(1)
dat_plant_2_temp2["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_2_agg_mst["DATE_TIME"]) + pd.DateOffset(2)
dat_plant_2_temp3["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_2_agg_mst["DATE_TIME"]) + pd.DateOffset(3)
dat_plant_2_temp4["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_2_agg_mst["DATE_TIME"]) + pd.DateOffset(4)
dat_plant_2_temp5["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_2_agg_mst["DATE_TIME"]) + pd.DateOffset(5)
dat_plant_2_temp6["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_2_agg_mst["DATE_TIME"]) + pd.DateOffset(6)
dat_plant_2_temp7["DATE_TIME"] = pd.DatetimeIndex(dat_plant_power_2_agg_mst["DATE_TIME"]) + pd.DateOffset(7)

dat_plant_power_2_agg_mst = pd.merge(dat_plant_power_2_agg_mst, dat_plant_2_temp1, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_2_agg_mst = pd.merge(dat_plant_power_2_agg_mst, dat_plant_2_temp2, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_2_agg_mst = pd.merge(dat_plant_power_2_agg_mst, dat_plant_2_temp3, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_2_agg_mst = pd.merge(dat_plant_power_2_agg_mst, dat_plant_2_temp4, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_2_agg_mst = pd.merge(dat_plant_power_2_agg_mst, dat_plant_2_temp5, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_2_agg_mst = pd.merge(dat_plant_power_2_agg_mst, dat_plant_2_temp6, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_power_2_agg_mst = pd.merge(dat_plant_power_2_agg_mst, dat_plant_2_temp7, left_on='DATE_TIME', right_on='DATE_TIME')

dat_plant_power_2_agg_mst["HOUR"] = dat_plant_power_2_agg_mst['DATE_TIME'].dt.hour
dat_plant_power_2_agg_mst["MINUTE"] = dat_plant_power_2_agg_mst['DATE_TIME'].dt.minute
dat_plant_weather_2_agg_mst = dat2.groupby('DATE_TIME').agg({
                                                    'AMBIENT_TEMPERATURE': 'mean',
                                                    'MODULE_TEMPERATURE': 'mean',
                                                    'IRRADIATION': 'mean'
                                                }).reset_index()

dat_plant_2_weather_agg_temp1 = dat_plant_weather_2_agg_mst.copy()
dat_plant_2_weather_agg_temp2 = dat_plant_weather_2_agg_mst.copy()
dat_plant_2_weather_agg_temp3 = dat_plant_weather_2_agg_mst.copy()
dat_plant_2_weather_agg_temp4 = dat_plant_weather_2_agg_mst.copy()
dat_plant_2_weather_agg_temp5 = dat_plant_weather_2_agg_mst.copy()
dat_plant_2_weather_agg_temp6 = dat_plant_weather_2_agg_mst.copy()
dat_plant_2_weather_agg_temp7 = dat_plant_weather_2_agg_mst.copy()

dat_plant_2_weather_agg_temp1["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_2_agg_mst["DATE_TIME"]) + pd.DateOffset(1)
dat_plant_2_weather_agg_temp2["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_2_agg_mst["DATE_TIME"]) + pd.DateOffset(2)
dat_plant_2_weather_agg_temp3["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_2_agg_mst["DATE_TIME"]) + pd.DateOffset(3)
dat_plant_2_weather_agg_temp4["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_2_agg_mst["DATE_TIME"]) + pd.DateOffset(4)
dat_plant_2_weather_agg_temp5["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_2_agg_mst["DATE_TIME"]) + pd.DateOffset(5)
dat_plant_2_weather_agg_temp6["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_2_agg_mst["DATE_TIME"]) + pd.DateOffset(6)
dat_plant_2_weather_agg_temp7["DATE_TIME"] = pd.DatetimeIndex(dat_plant_weather_2_agg_mst["DATE_TIME"]) + pd.DateOffset(7)

dat_plant_weather_2_agg_mst = pd.merge(dat_plant_weather_2_agg_mst, dat_plant_2_weather_agg_temp1, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_2_agg_mst = pd.merge(dat_plant_weather_2_agg_mst, dat_plant_2_weather_agg_temp2, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_2_agg_mst = pd.merge(dat_plant_weather_2_agg_mst, dat_plant_2_weather_agg_temp3, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_2_agg_mst = pd.merge(dat_plant_weather_2_agg_mst, dat_plant_2_weather_agg_temp4, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_2_agg_mst = pd.merge(dat_plant_weather_2_agg_mst, dat_plant_2_weather_agg_temp5, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_2_agg_mst = pd.merge(dat_plant_weather_2_agg_mst, dat_plant_2_weather_agg_temp6, left_on='DATE_TIME', right_on='DATE_TIME')
dat_plant_weather_2_agg_mst = pd.merge(dat_plant_weather_2_agg_mst, dat_plant_2_weather_agg_temp7, left_on='DATE_TIME', right_on='DATE_TIME')

dat_plant_2_mst = pd.merge(dat_plant_power_2_agg_mst, dat_plant_weather_2_agg_mst, left_on='DATE_TIME', right_on='DATE_TIME')
dat_7d = pd.concat([dat_plant_1_mst, dat_plant_2_mst])
dat_7d.head()
def kfolds_rmse(data, digits, model = 'RFR'):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    rmse = []
    
    train_feature = data.drop(["DATE_TIME","TOTAL_YIELD"],axis = 1).columns
    
    for digit in digits:
        train = data[data.index%10 != digit]
        test = data[data.index%10 == digit]
        if model == 'LR':
            model = LinearRegression()
            reg = model.fit(train[train_feature], train["TOTAL_YIELD"])

        elif model == 'RFR':
            model = RandomForestRegressor()
            reg = model.fit(train[train_feature], train["TOTAL_YIELD"])
            
        pred = reg.predict(test[train_feature])
        rmse.append(mean_squared_error(test["TOTAL_YIELD"], pred))          
    return rmse
lr_3d = kfolds_rmse(dat_3d, [0,1,2,3,4,5,6,7,8,9], "LR")
print("Linear Regression with 3 Days history : ", lr_3d)
rfr_3d = kfolds_rmse(dat_3d, [0,1,2,3,4,5,6,7,8,9], "RFR")
print("Random Forest Regression with 3 Days history : ", kfolds_rmse(dat_3d, [0,1,2,3,4,5,6,7,8,9], "RFR"))
import matplotlib.pyplot as plt
plt.plot(lr_3d, label = "Linear Regression")
plt.plot(rfr_3d, label = "Random Forest Regression")
plt.xlabel('x - axis')
plt.ylabel('RMSE')
plt.legend()
plt.show()
lr_7d = kfolds_rmse(dat_7d, [0,1,2,3,4,5,6,7,8,9], "LR")
print("Linear Regression with 7 Days history : ", lr_7d)
rfr_7d = kfolds_rmse(dat_7d, [0,1,2,3,4,5,6,7,8,9], "RFR")
print("Random Forest Regression with 7 Days history : ", rfr_7d)
