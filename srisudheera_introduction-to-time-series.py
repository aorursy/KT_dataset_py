import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
data=pd.read_csv('../input/international-airline-passengers.csv')
train = data[:int(0.7*(len(data)))]
valid = data[int(0.7*(len(data))):]
#preprocessing (since arima takes univariate series as input)
train.drop('Month',axis=1,inplace=True)
valid.drop('Month',axis=1,inplace=True)
#plotting the data
import matplotlib.pyplot as plt
train['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60'].plot()
valid['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60'].plot()
#building the model
from pyramid.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)

forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()
#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(valid,forecast))
print(rms)