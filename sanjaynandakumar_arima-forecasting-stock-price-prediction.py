# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import pandas as pd
import numpy as np
import os
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
warnings.filterwarnings('ignore')
path = "../input/data_stocks.csv"
data = pd.read_csv(path)
data.replace([np.inf, -np.inf], np.nan)
data.dropna(inplace=True)
print("Number of Records in input data : ",len(data["DATE"]))
data.head(5)
#Stock analysis of NASDAQ.AAPL
stockname="NASDAQ.AAPL"
from datetime import datetime
date=pd.date_range("2000-01-01", periods=100,freq='B').to_pydatetime().tolist()
date=pd.DataFrame(date,columns=["DATE"])
stock=pd.DataFrame()
stock=pd.DataFrame(data[stockname][41166:41266])
date=date.reset_index(drop=True)
stock=stock.reset_index(drop=True)
aapl=pd.concat([date,stock],axis=1)
aapl.info()
aapl.DATE = pd.to_datetime(aapl.DATE)
aapl.set_index('DATE', inplace=True)
print(aapl.head(5))
print(aapl.tail(5))
aapl.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
aapl.describe().T
#Differencing Differencing is a decomposition process through which trend and seasonality are eliminated. Here, we usually take the difference of observation with particular instant with previous instant.

#First-order differencing
aapl.diff().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Year', fontsize=20)
# Comulative Return
dr = aapl.cumsum()
dr.plot()
plt.title('AAPL Cumulative Returns')
"""
Auto regression The approach of auto regression is basically an extended version of simple and multiple linear regressions. The only major difference is that the predictive relationship between the dependent and independent variables are assumed with in the previous values of same time series.
As a proper definition we can say that “A statistical model is said to be autoregressive if it predicts future values based on previous values.” Although there is no proved evidence that this has worked out most of the times in history, there is a major application of auto regression is still processed in financial markets and weather forecasting. 
The number of predictor variables is usually determined by the statistician which is nothing but the count of previous values that the asset had as a market or intrinsic measure. 
Mathematically, it is denoted as – AR(parameter) where parameter is the number of independent variables or the count of past values considered for forecasting. 
For example, AR (1) means that it is an autoregressive process where the immediate single past value in a defined time period is considered as independent variable. 
AR (2) means that it is an autoregressive process where the immediate two past values in a defined time period is considered as independent variables. AR (n) means that it is an autoregressive process where the immediate n past values in a defined time period is considered as independent variables. 
AR (0) is the easiest form of regression where the future value is considered as similar to the current value but this technically does not holds in growing or depreciating asset. 
We can define AR(4) mathematically as, Yt = B1 (Yt-1) + B2 (Yt-2) + B3 (Yt-3) + B4 (Yt-4) Where, Yt = Predicted value for next time period. Yt-1 = Recorded value for immediate past time period. Yt-2= Recorded value for immediate second past time period. Yt-1 = Recorded value for immediate third past time period. Yt-1 = Recorded value for immediate fourth past time period. B1 = Coefficient of auto regression for immediate past time period value. B2 = Coefficient of auto regression for immediate second past time period value. B3 = Coefficient of auto regression for immediate third past time period value. B4 = Coefficient of auto regression for immediate fourth past time period value.
"""
plt.figure(figsize=(10,10))
lag_plot(aapl[stockname], lag=5)
plt.title('AAPL Autocorrelation plot')
# import the plotting functions for act and pacf  
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(aapl[stockname], lags=5)
plot_pacf(aapl[stockname], lags=5)
train_data, test_data = aapl[0:int(len(aapl)*0.8)], aapl[int(len(aapl)*0.8):]
plt.figure(figsize=(12,7))
plt.title('AAPL Prices')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.plot(aapl[stockname], 'blue', label='Training Data')
plt.plot(test_data[stockname], 'green', label='Testing Data')
plt.legend()
def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))

train_ar = train_data[stockname].values
test_ar = test_data[stockname].values

history = [x for x in train_ar]
print(type(history))
predictions = list()
for t in range(len(test_ar)):
    model = ARIMA(history, order=(5,1,2))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
    #print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test_ar, predictions)
print('Testing Mean Squared Error: %.3f' % error)
error2 = smape_kun(test_ar, predictions)
print('Symmetric mean absolute percentage error: %.3f' % error2)
plt.figure(figsize=(12,7))
plt.plot(aapl[stockname],'green', color='blue', label='Training Data')
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', 
         label='Predicted Price')
plt.plot(test_data.index, test_data[stockname], color='red', label='Actual Price')
plt.title('AAPL Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.legend()
plt.figure(figsize=(12,7))
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', 
         label='Predicted Price')
plt.plot(test_data.index, test_data[stockname], color='red', label='Actual Price')
plt.title('AAPL Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.legend()

actual=pd.DataFrame()
actual=pd.DataFrame(test_ar,columns=["Actual"])
predicted=pd.DataFrame(list(predictions),columns=["Predicted"])
actual=actual.reset_index(drop=True)
predicted=predicted.reset_index(drop=True)
output=pd.concat([actual,predicted],axis=1)
print(output.head(10))

