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
from fbprophet import Prophet
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
df['Month']= pd.to_datetime(df['Month'])
df = df.rename(columns={ "Month" : "ds" , "Passengers" : "y" })
df.head()
model = Prophet()
model.fit(df)
#predict for the next 10 months

future = model.make_future_dataframe(periods= 10, freq= 'M')
forecast = model.predict(future)
forecast.head()
forecast[['ds' , 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']]
#yhat is the prediction while yhat_lower and yhat_upper are the upper and lower boundaries 
model.plot(forecast)
plt.show()
forecast
plt.figure(figsize=(12 , 7))
plt.plot(forecast.ds, forecast.trend)
plt.grid()
plt.xlabel('ds')
plt.ylabel('trend')
plt.figure(figsize=(12 , 7))
plt.plot(forecast.ds, forecast.yearly)
plt.grid()
plt.xlabel('ds')
plt.ylabel('yearly')
