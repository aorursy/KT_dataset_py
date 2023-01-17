#### import the libraries we need

import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
### read dataset

data = pd.read_csv("../input/license_plates_acution_data.csv",usecols=["Date",

                                                                          "num_bidder",

                                                                          "num_plates",

                                                                          "lowest_deal_price",

                                                                          "avg_deal_price"],dtype={

    "Date":"str",

    "num_bidder":"float",

    "num_plates":"float",

    "lowest_deal_price":"float",

    "avg_deal_price":"float"

})

data.head()
### convert Date column into datetime and set it as index for time series analysis

data["Date"] = pd.to_datetime(data["Date"])

data.set_index("Date",inplace=True)

data.head()
#### check missing values

for y in range(2002,2018):

    print(y,": ", data[str(y)].shape)
#dig into 2008, Feb is missing,  fill in missing row with mean of each column 

feb_2008 = pd.DataFrame([data["2008"].mean(axis=0)],index=[pd.to_datetime("2008-02-01")],columns=data.columns)

data = pd.concat([data,feb_2008])

data.sort_index(inplace=True)

data["2008"]
#### we are interested in predicting future lowest_deal_price, let's plot the column to see what it looks like

plt.figure(figsize=(8,8))

data["lowest_deal_price"].plot()

plt.title("Lowest Deal Price")
### clearly there is a trend going upwards, but there are also some outliers around 2003,2008,2001

### the outliers were probably caused by governmental interference or policy change, which is unusual

### let's deal with these ourliers by replacing those values with means of the respective year

data.set_value("2002-12-01","lowest_deal_price",data["2002"]["lowest_deal_price"].mean(axis=0))

data.set_value("2008-01-01","lowest_deal_price",data["2007-02":"2008-02"]["lowest_deal_price"].mean(axis=0))

data.set_value("2010-12-01","lowest_deal_price",data["2010"]["lowest_deal_price"].mean(axis=0))

### now it looks smoothier

data["lowest_deal_price"].plot()
#### General steps of time series analysis:

#### 1. visuliaze the time series

#### 2. stationize the time series

#### 3. plot acf and pacf to find optimal parameters for the model

#### 4. build ARIMA model

#### 5. make predictions



#### now let's build a function to test stationarity

def test_stationary(ts_data):

    ## visually check

    rol_mean = pd.rolling_mean(ts_data,window=12)

    rol_std = pd.rolling_std(ts_data,window=12)

    plt.figure(figsize=(6,6))

    plt.plot(ts_data,label='original')

    plt.plot(rol_mean,label="rolling mean")

    plt.plot(rol_std,label="rolling std")

    plt.legend()

    ## adfuller test

    result = adfuller(ts_data,autolag="AIC")

    print("---"*5,"ADF TEST RESULT","---"*5,"\n")

    print(" ADF value: ",result[0])

    print(" P_Value: ",result[1])

    print(" Lags used: ", result[2])

    print(" num of obs: ", result[3])

    print("intervals: ", result[4])
# stationarity check of the original data



### from the result( p_value), we can see that the original time series are not stationary

test_stationary(data["lowest_deal_price"])
### let's decompose the trend, seasonal components from the data

decomposition = seasonal_decompose(data["lowest_deal_price"])

trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid

plt.figure(figsize=(10,12))

plt.subplot(411)

plt.plot(data["lowest_deal_price"])

plt.title("Original")

plt.subplot(412)

plt.plot(trend)

plt.title("Trend")

plt.subplot(413)

plt.plot(seasonal)

plt.title("Seasonal")

plt.subplot(414)

plt.plot(residual)

plt.title("Residual")

plt.tight_layout()
### now let's check stationarity of the residuls

#### note that we have to drop Na first as it has been differenced

test_stationary(residual.dropna())
### Also we can try to manually stationize it, this can usually be done by log transformation or difference

### let's try log transformation first

data_log = np.log(data["lowest_deal_price"])

test_stationary(data_log)

#### it looks like log transformation failed to do the trick

#### let's try difference it once

### pandas has this built-in function diff() to do the differencing

### or you can do : data - data.shift()

data_diff_1 = data["lowest_deal_price"].diff().dropna()

test_stationary(data_diff_1)
### it worked, and our time series is stationary now.

### let's plot the acf and pacf for model selection

acf_plot = plot_acf(data_diff_1,lags=20)

pacf_plot = plot_pacf(data_diff_1,lags=20)
### from the acf and pacf, it looks like we need a ARIMA model of p=1,d=1,q=1

model = ARIMA(data["lowest_deal_price"],order=(1,1,1))

model_fit = model.fit(disp=1)

summary = model_fit.summary()

fitted_values = model_fit.fittedvalues

print(summary)

plt.figure(figsize=(10,5))

plt.plot(data_diff_1,label="Original")

plt.plot(fitted_values, label="Model values")

plt.legend()
### it looks like the model is doing alright

### this will be the benchmark for model modification and improvement

### some of the future steps to take would be improving models and making predictions