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
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv("../input/into-the-future/train.csv")
test = pd.read_csv("../input/into-the-future/test.csv") 
train.head()
test.head()
train = pd.read_csv("../input/into-the-future/train.csv", index_col=[1], parse_dates=True, squeeze=True)
test = pd.read_csv("../input/into-the-future/test.csv", index_col=[1], parse_dates=True, squeeze=True)
train.shape
test.shape
train.head()
test.head()
plt.plot(train['feature_1'], train['feature_2'])
#plt.plot(train['id'], train['feature_2'])
plt.scatter(train['feature_1'], train['feature_2'])
train.describe()
train.isnull().sum()
train[['feature_1','feature_2']].corr()
x = train[['feature_1']]
y = train[['feature_2']]
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)
pred = reg.predict(test[['feature_1']])
pred[:10]
test['predicted_feature_2'] = pred
test.head()
plt.scatter(test['feature_1'], test['predicted_feature_2'])
data = pd.concat([train['feature_2'],test['predicted_feature_2']], axis=0)
data.head()
data.plot()
data_ma = data.rolling(window=10).mean()
data_ma.plot()
data_base = pd.concat([data, data.shift(1)], axis=1)
data_base.head()
data_base.columns = ['Actual', 'Forecast']
data_base.dropna(inplace=True)
data_base.head()
from sklearn.metrics import mean_squared_error
import numpy as np
data_error = mean_squared_error(data_base['Actual'], data_base['Forecast'])
np.sqrt(data_error)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(data)
# Q=30, P =2, d=0-2
plot_pacf(data)
from statsmodels.tsa.arima_model import ARIMA
data_train = data[:564]
data_test = data[564:]
data_model = ARIMA(data_train, order=(30,2,1))
data_model_fit = data_model.fit()
data_model_fit.aic
data_forecast = data_model_fit.forecast(steps=564/2)[0]
np.sqrt(mean_squared_error(data_test, data_forecast))