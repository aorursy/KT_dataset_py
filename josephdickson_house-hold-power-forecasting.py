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
#first, we read the data and preprocess it in a csv
data_path = "../input/house-hold-power-consumption/household_power_consumption.txt"

#read data
data = pd.read_csv(data_path, delimiter=";", parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False, na_values=['nan', '?'], index_col='datetime')
data.head()
data.isnull().sum()
data.shape
# lets see the type of timeserries data we have before proceeding to handle the missing data
test_data = data.copy()
test_data.dropna(inplace=True)
test_data.isnull().sum()
test_data = test_data.assign(Global_active_power=test_data.Global_active_power.fillna(test_data.Global_active_power.mean()))
test_data = test_data.assign(Global_reactive_power=test_data.Global_reactive_power.fillna(test_data.Global_reactive_power.mean()))
test_data = test_data.assign(Voltage = test_data.Voltage.fillna(test_data.Voltage.mean()))
test_data = test_data.assign(Global_intensity = test_data.Global_intensity.fillna(test_data.Global_intensity.mean()))
test_data = test_data.assign(Sub_metering_1 = test_data.Sub_metering_1.fillna(test_data.Sub_metering_1.mean()))
test_data = test_data.assign(Sub_metering_2 = test_data.Sub_metering_2.fillna(test_data.Sub_metering_2.mean()))
test_data = test_data.assign(Sub_metering_3 = test_data.Sub_metering_3.fillna(test_data.Sub_metering_3.mean()))
test_data.isna().sum()
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(test_data.index, test_data.Global_active_power, "--", marker="*")
plt.grid()
plt.xlabel('date')
plt.ylabel('Global_active_power')
test_data.Global_active_power.resample('D').sum().plot()

#we also do some smoothening to better understand the progress in our dataset
test_data.Global_reactive_power.resample('D').sum().plot()
plt.figure(figsize=(10,6))
plt.plot(test_data.index, test_data.Global_active_power, "--", marker="*")
plt.grid()
plt.xlabel('year')
plt.ylabel('Passengers')
daily_data = test_data.resample('D').sum()

daily_data = daily_data.reset_index()
daily_data.head()
daily_data.drop(columns=["Global_reactive_power","Voltage","Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"], inplace=True)
daily_data.drop(columns=["index"], inplace=True)
daily_data
daily_data["datetime"] = pd.to_datetime(daily_data["datetime"])
daily_data.rename(columns={"datetime": "ds", "Global_active_power":"y"}, inplace=True)
daily_data.head()
#divide into train and test set

X_train = daily_data.iloc[:365, :]
X_test = daily_data.iloc[-365:, :]
X_train.shape
X_test.shape
daily_data.head()
daily_data.tail()
X_test.tail()
from fbprophet import Prophet

model2 = Prophet()
model2.fit(X_train)

#predict for next 10 months
future = model2.make_future_dataframe(periods=12, freq="D")
forecast = model2.predict(future)
forecast.head()
forecast.tail()
365 * 3


pred = pd.DataFrame(forecast.ds, forecast.yhat)
model2.plot(forecast)
def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100
new_data = test_data
new_data.head()
new_data.reset_index(inplace=True)
new_data.head()
new_data["datetime"] = pd.to_datetime(new_data["datetime"])
new_data.rename(columns={"datetime": "ds", "Global_reactive_power":"add1","Voltage": "add2", "Global_intensity":"add3", "Sub_metering_1":"add4", "Sub_metering_2":"add5", "Sub_metering_3":"add6","Global_active_power":"y"}, inplace=True)
new_model =Prophet()
#divide into train and test set

X_train_new = new_data.iloc[:365, :]
X_test_new = new_data.iloc[-365:, :]
new_model.fit(X_train_new)

#predict for next 10 months
future = new_model.make_future_dataframe(periods=10, freq="D")
forecast = new_model.predict(future)
forecast.head()
new_model.plot(forecast)
