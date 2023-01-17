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
df=pd.read_csv("/kaggle/input/sarima/airline_passengers (1).csv",parse_dates=True,index_col="Month")
df.head()
df.plot()
from statsmodels.tsa.seasonal import seasonal_decompose
result=seasonal_decompose(df)
result.plot();
df["MA-6"]=df["Thousands of Passengers"].rolling(6).mean()
df["MA-12"]=df["Thousands of Passengers"].rolling(12).mean()
df.head()
df.plot()
df["ewm-0.4"]=df["Thousands of Passengers"].ewm(alpha=0.4).mean()
df["ewm-0.8"]=df["Thousands of Passengers"].ewm(alpha=0.8).mean()

df[["Thousands of Passengers","ewm-0.4","ewm-0.8"]].plot()
from statsmodels.tsa.api import SimpleExpSmoothing
fit_1=SimpleExpSmoothing(df["Thousands of Passengers"]).fit(smoothing_level=0.2,optimized=False)
forecastm1=fit_1.forecast(12).rename(r'$\alpha-0.2$')
forecastm1.plot()
fit_2=SimpleExpSmoothing(df["Thousands of Passengers"]).fit(smoothing_level=0.4,optimized=False)
forecastm2=fit_2.forecast(12).rename(r'$\alpha-0.6$')
forecastm2.plot()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df["Thousands of Passengers"],lags=100);
plot_pacf(df["Thousands of Passengers"]);
from statsmodels.tsa.stattools import adfuller
help(adfuller)
adfuller(df["Thousands of Passengers"])
def adf_test(data):
    test_output=adfuller(data)
    if test_output[1]<=0.05:
        print("Stationary")
    else:
        print("non-stationary")
    
adf_test(df["Thousands of Passengers"])
df["Thousands of Passengers"].plot()
df["Seasonal Diff"]=df["Thousands of Passengers"]-df["Thousands of Passengers"].shift(12)
adfuller(df["Seasonal Diff"].dropna())
adf_test(df["Seasonal Diff"].dropna())
df["Seasonal Diff"].plot()
plot_acf(df["Seasonal Diff"].dropna());
plot_pacf(df["Seasonal Diff"].dropna());
df.index.freq="MS"
import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(df["Thousands of Passengers"],order=(2,0,0),seasonal_order=(0,1,0,12))
re=model.fit()
re.summary()
len(df)
df["fore"]=re.predict(start=132,end=144,dynamic=True)
df[["fore","Thousands of Passengers"]].plot()
df.tail()
from pandas.tseries.offsets import DateOffset
fut_date=[df.index[-1]+DateOffset(months=x) for x in range(0,13)]
fut_date
fut_df=pd.DataFrame(index=fut_date[1:],columns=df.columns)
fut_df.head()
fut_df=pd.concat([df,fut_df])
fut_df.tail()
fut_df["fore"]=re.predict(start=145,end=156,dynamic=True)
fut_df[["Thousands of Passengers","fore"]].plot()
fut_df.tail()
