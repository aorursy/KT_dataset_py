# import os
# print(os.listdir("../input/new_case.xlsx"))
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_excel('../input/new_case.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index(['Date'])
data

# SPLITING DATA INOT TRAIN AND TEST . HERE TELL WILL CONTAIN NAN FIELDS
train = data[:97]
valid = data[97:]

# # #plotting the data
train['Total Confirmed'].plot()
valid['Total Confirmed'].plot()
train
valid
!pip install pmdarima
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
# Fit your model
model = pm.auto_arima(train, seasonal=True, m=6)
# FORECASTING
forecasts = model.predict(valid.shape[0])  # predict N steps into the future
# Visualize the forecasts (blue=train, green=forecasts)
x = np.arange(data.shape[0])
plt.figure(figsize=(12,10))
plt.plot(x[:97], train, c='blue', label='ACTUAL')
plt.plot(x[97:], forecasts, c='green', label='PREDICTED')

plt.legend(loc='best')
plt.show()
for i in forecasts:
    print(i)