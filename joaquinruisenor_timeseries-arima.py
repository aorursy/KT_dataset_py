import pandas as pd
import numpy as np
import datetime
%matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
%matplotlib inline
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import string
from sklearn.metrics import mean_squared_error, mean_absolute_error
df_arima = pd.read_csv('../input/timeseries-prophet/Energy_PJM.csv')
df_arima.head()
plot_acf(df_arima["Energy"],alpha=0.1, lags=90)
plot_pacf(df_arima["Energy"],alpha=0.1, lags=20)
#We split the data into training and testing set
df_arima.set_index('Datetime', inplace = True) #make the index is out date time
split_date = '2017-01-01'
df_train = df_arima.loc[df_arima.index <= split_date].copy()
df_test = df_arima.loc[df_arima.index > split_date].copy()
df_train = df_train.reset_index()
df_test = df_test.reset_index()
df_train.head()
def run_arima_model(df, ts, p, d, q):
    
    from statsmodels.tsa.arima_model import ARIMA

    # fit ARIMA model on time series
    model = ARIMA(df[ts], order=(p, d, q))  
    results_ = model.fit(disp=-1)  
  
    # get lengths correct to calculate RSS
    len_results = len(results_.fittedvalues)
    ts_modified = df[ts][-len_results:]
  
    # calculate root mean square error (RMSE) and residual sum of squares (RSS)
    rss = sum((results_.fittedvalues - ts_modified)**2)
    rmse = np.sqrt(rss / len(df[ts]))
  
    # plot fit
    plt.plot(df[ts])
    plt.plot(results_.fittedvalues, color = 'red')
    plt.title('For ARIMA model (%i, %i, %i) for ts %s, RSS: %.4f, RMSE: %.4f' %(p, d, q, ts, rss, rmse))
  
    plt.show()
    plt.close()
  
    return results_ 
df_modelo = df_arima.reset_index()
df_arima.head().append(df_arima.tail())
# AR model with 1st order differencing - ARIMA (1,0,0) or AR model
model_AR = run_arima_model(df = df_modelo,  
                           ts = 'Energy',
                           p = 1, 
                           d = 0, 
                           q = 0)

# MA model with 1st order differencing - ARIMA (0,0,1) or MA model
model_MA = run_arima_model(df = df_modelo, 
                           ts = 'Energy',
                           p = 0, 
                           d = 0, 
                           q = 1)

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100