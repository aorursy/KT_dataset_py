# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# imoprt the dataset
df = pd.read_csv('../input/sales-of-shampoo/sales-of-shampoo-over-a-three-ye.csv')
df.head()
df.shape
df.dtypes
df = pd.read_csv('../input/sales-of-shampoo/sales-of-shampoo-over-a-three-ye.csv', parse_dates=True,index_col=[0],squeeze=True)
df.head()
type(df)
# we will visaulise our data
df.plot()
df.plot(style = 'k.')
df_ma = df.rolling(window=10).mean()
df_ma
# we will now plot the smoothed dataframe.
df_ma.plot()
# we will now compare the smoothed data to the original data.
df.plot()
df_ma.plot()
df
df_naive = pd.concat([df,df.shift(1)],axis=1)
df_naive
df_naive.dropna(inplace=True)
df_naive.columns = ['Actual Data','Shifted Data']
df_naive 
# we will get the mean_sqaured_error
from sklearn.metrics import mean_squared_error
error_1 = mean_squared_error(df_naive['Actual Data'],df_naive['Shifted Data'])
error_1
error_1 = np.sqrt(error_1)
error_1
# acf and pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# Here we are considering the original dataframe, our original dataframe consisted NaN value in the last row so we dropped it.
df.dropna(inplace=True)
# plot_acf will give us the q value(Autocorrelation)

plot_acf(df)
# plot_pacf will give us the q value(Partial Autocorrelation)

plot_pacf(df)
# p = 2
# d = 1
# q = 3
# we will now create our ARIMA model.
from statsmodels.tsa.arima_model import ARIMA
# we will now create a training and testing data 
df_train = df[:26]
df_test = df[26:]
df_train.shape,df_test.shape
model = ARIMA(df_train, order=(3,1,2))
model_fit = model.fit()
model_fit.aic
model_forecast = model_fit.forecast(steps=10)[0]
error2 = np.sqrt(mean_squared_error(df_test,model_forecast))
error2
import warnings
warnings.filterwarnings("ignore")
p_values = range(0,5)
d_values = range(0,3)
q_values = range(0,5)
for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p,d,q)
            train,test = df[0:26], df[26:]
            predictions = []
            for i in range(len(test)):
                try:
                    model = ARIMA(train, order)
                    model_fit = model.fit(disp=0)
                    pred_y = model_fit.forecast()[0]
                    predictions.append(pred_y)
                    error = np.sqrt(mean_squared_error(test,predictions))
                    print('Arima%s RMSE = %.2f'% (order,error))
                except:
                    continue
