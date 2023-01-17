import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



fbirth= pd.read_csv("../input/female-births-in-ca/daily-total-female-births-ca.csv",index_col=[0],parse_dates=[0],squeeze =True)
fbirth.head()
type(fbirth)
series_value = fbirth.values

type(series_value)
fbirth.size
fbirth.tail()
fbirth.describe()
fbirth.plot()
fbirth_mean= fbirth.rolling(window=20).mean()
fbirth_mean.plot()

fbirth.plot()

fbirth_mean.plot()
value = pd.DataFrame(series_value)
birth_df = pd.concat([value,value.shift(1)],axis=1)
birth_df
birth_df.columns=['Actual_birth','Forecast_birth']
birth_df.head()
from sklearn.metrics import mean_squared_error

import numpy as np
birth_test = birth_df[1:]

birth_test.head()
birth_test.tail()
birth_error= mean_squared_error(birth_test.Actual_birth,birth_test.Forecast_birth)
birth_error
# an error of 9 approx



np.sqrt(birth_error) 
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf



#plot_acf is used to identify parameter Q

#ARIMA(p,d,q)



plot_acf(fbirth)
plot_pacf(fbirth) # to identify value of p
# p = 2,3 and q = 3,4  and d=0



birth_train = fbirth[0:330]

birth_test = fbirth[330:365]
from statsmodels.tsa.arima_model import ARIMA
birth_model = ARIMA(birth_train,order=(2,1,3))
birth_model_fit = birth_model.fit()
birth_model_fit.aic
birth_forecast= birth_model_fit.forecast(steps=35)[0]
birth_forecast
birth_test
np.sqrt(mean_squared_error(birth_test,birth_forecast))