# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import some libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
# !pip3 install fbprophet
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
block_0_df = pd.read_csv("../input/halfhourly_dataset/halfhourly_dataset/block_0.csv")
block_0_df.head()
block_0_df.dtypes
# set tstp to index with datetime type
block_0_df = block_0_df.set_index("tstp")
block_0_df.index = block_0_df.index.astype("datetime64")
# set energy consumption data to float type
block_0_df = block_0_df[block_0_df["energy(kWh/hh)"] != "Null"]
block_0_df["energy(kWh/hh)"] = block_0_df["energy(kWh/hh)"].astype("float64")
# Choose only 1 house by LCLid "MAC000002"
block_0_df = block_0_df[block_0_df["LCLid"] == "MAC000002" ]
# plot energy consumption data with dataframe module
block_0_df.plot(y="energy(kWh/hh)", figsize=(12, 4))
train_size = int(0.8 * len(block_0_df))
X_train, X_test = block_0_df[:train_size].index, block_0_df[train_size:].index
y_train, y_test = block_0_df[:train_size]["energy(kWh/hh)"].values, block_0_df[train_size:]["energy(kWh/hh)"].values
train_df = pd.concat([pd.Series(X_train), pd.Series(y_train)], axis=1, keys=["ds", "y"])
test_df = pd.concat([pd.Series(X_test), pd.Series([0]*len(y_test))], axis=1, keys=["ds", "y"])
answer_df = pd.concat([pd.Series(X_test), pd.Series(y_test)], axis=1, keys=["ds", "y"])
# make model with Prophet by Facebook
model = Prophet()
model.fit(train_df)
forecast = model.predict(test_df)
forecast.head()
model.plot(forecast)
# with plot_components method, we can visualize the data components
model.plot_components(forecast)
# Analysis with cross validation method
# This cell takes some minutes.
df_cv = cross_validation(model, horizon="60 days")
df_cv.head()
# With performance_metrics, we can visualize the score
df_p = performance_metrics(df_cv)
df_p
plt.plot(answer_df["ds"], answer_df["y"])
plt.plot(forecast["ds"], forecast["yhat"])
# We have to add week and month seasonality
model = Prophet(weekly_seasonality=True)
model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
model.fit(train_df)
forecast = model.predict(test_df)
forecast.head()
model.plot(forecast)
model.plot_components(forecast)
# add daily seasonality
model = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=True, growth="logistic")
model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
train_df["cap"] = 0.25
train_df["floor"] = 0.1
model.fit(train_df)
test_df["cap"] = 0.25
test_df["floor"] = 0.1
forecast = model.predict(test_df)
forecast.head()
model.plot(forecast)
model.plot_components(forecast)
df_cv = cross_validation(model, horizon="60 days")
df_cv.head()
df_p = performance_metrics(df_cv)
df_p
plt.figure(figsize=(12, 4))
plt.plot(answer_df["ds"], answer_df["y"])
plt.plot(forecast["ds"], forecast["yhat"])
holiday_df = pd.read_csv("../input/uk_bank_holidays.csv", names=("ds", "holiday"), header=0)
holiday_df.head()
# add daily seasonality
model = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=True, 
                growth="logistic", holidays=holiday_df, changepoint_prior_scale=0.2)
model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
train_df["cap"] = 0.25
train_df["floor"] = 0.1
model.fit(train_df)
test_df["cap"] = 0.25
test_df["floor"] = 0.1
forecast = model.predict(test_df)
forecast.head()
model.plot(forecast)
model.plot_components(forecast)
df_cv = cross_validation(model, horizon="60 days")
df_cv.head()
df_p = performance_metrics(df_cv)
df_p
plt.figure(figsize=(12, 4))
plt.plot(answer_df["ds"], answer_df["y"])
plt.plot(forecast["ds"], forecast["yhat"])
