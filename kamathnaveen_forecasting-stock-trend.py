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
import numpy as np
import pandas as pd
from pandas_datareader import data
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
stock_symbol = 'AMT'
stk = data.DataReader(stock_symbol,start = '2000-01-01', end='2020-06-10',data_source='yahoo')['Adj Close']
stock_df = pd.DataFrame(stk)
stock_df = stock_df.resample('W').last()
stock_df.tail()
model = ARIMA(stock_df,order=(1,1,1))

model_fit = model.fit()
print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind ='kde')
plt.show()
print(residuals.describe())
#Forecast

mod = ARIMA(stock_df,order=(1,1,1))
res = mod.fit()
plt.rcParams["figure.figsize"] = (20,10)
res.plot_predict(start='2015', end='2025')
plt.legend(fontsize=10)
plt.show()
