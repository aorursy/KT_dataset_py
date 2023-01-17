# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import mean_squared_error
import numpy as np

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
f_birth = pd.read_csv('/kaggle/input/daily-total-female-births-in-california-1959/daily-total-female-births-CA.csv')
f_birth = pd.read_csv('/kaggle/input/daily-total-female-births-in-california-1959/daily-total-female-births-CA.csv', index_col=[0],parse_dates=[0])
f_birth.head()
f_birth.describe()
series_value = f_birth.values
f_birth.plot()
f_birth_mean = f_birth.rolling(window = 25).mean()
f_birth.plot()
f_birth_mean.plot()
value = pd.DataFrame(series_value)
birth_df= pd.concat([value,value.shift(1)],axis = 1)
birth_df.columns = ['Actual_birth','Forecast_birth']
birth_df.head()
birth_test = birth_df[1:]
birth_test.head(2)
birth_error = mean_squared_error(birth_test.Actual_birth, birth_test.Forecast_birth)
birth_error
np.sqrt(birth_error)
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(f_birth)
plot_pacf(f_birth)
birth_train = f_birth[0:330]
birth_test = f_birth[330:365]
from statsmodels.tsa.arima_model import ARIMA
birth_model = ARIMA(birth_train, order=(2,1,3))

birth_model_fit = birth_model.fit()
birth_model_fit.aic
birth_forecast = birth_model_fit.forecast(steps = 35)[0]
birth_forecast
np.sqrt(mean_squared_error(birth_test,birth_forecast))
