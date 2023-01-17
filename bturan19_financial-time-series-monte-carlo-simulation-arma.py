# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data =pd.read_csv("../input/all_stocks_5yr.csv")
data.head()
data.tail()
#I just picked the first stock
data = data[data.Name == 'AAL']
data.tail()
from scipy.stats import norm
log_returns = np.log(1 + data.close.pct_change())
u = log_returns.mean() #Mean of the logarithmich return
var = log_returns.var() #Variance of the logarithic return
drift = u - (0.5 * var) #drift / trend of the logarithmic return
stdev = log_returns.std() #Standard deviation of the log return


t_intervals = 250 #I just wanted to forecast 250 time points
iterations = 10 #I wanted to have 10 different forecast

daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iterations)))
#daily_returns actually is some kind of a noise. When we multiply this with the t time price, we can obtain t+1 time price
S0 = data.close.iloc[-1]
S0
#Let us first create en empty matrix such as daily returns
price_list = np.zeros_like(daily_returns)
price_list[0] = S0
price_list
# With a simple for loop, we are going to forecast the next 250 days
for t in range(1, t_intervals):
    price_list[t] = price_list[t - 1] * daily_returns[t]
price_list = pd.DataFrame(price_list)
price_list['close'] = price_list[0]
price_list.head()
close = data.close
close = pd.DataFrame(close)
frames = [close, price_list]
monte_carlo_forecast = pd.concat(frames)
monte_carlo_forecast.head()
monte_carlo_forecast.tail()
monte_carlo = monte_carlo_forecast.iloc[:,:].values
import matplotlib.pyplot as plt
plt.figure(figsize=(17,8))
plt.plot(monte_carlo)
plt.show()
#Let's see the distribution of the log returns
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
#Thanks to https://www.kaggle.com/kanncaa1/plotly-tutorial-for-beginners 
trace = go.Histogram(x=log_returns,opacity=0.85,name = "Logarithmic Return", marker=dict(color='rgba(0, 0, 255, 0.8)'))
info = [trace]
layout = go.Layout(barmode='overlay',
                   title='Distribution of the Logarithmic Returns',
                   xaxis=dict(title='Logarithmic Return'),
                   yaxis=dict( title='Dist'),
)
fig = go.Figure(data=info, layout=layout)
iplot(fig)
#Note bad huh, it means that the return can be modeled with more traditional methods.
#I'm not going to analyse the stationarity or other hypothesis because I did some of them in my other kernel.
#In fact there are just so many hypothesis testing and I don't want to go into details of that.
data['log_return'] = np.log(1 + data.close.pct_change())
data.reset_index(inplace=True)
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data.head()
data = data.dropna()
#But we can examine the seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data.log_return, freq = 260) #Was there 260 workdays in a year?
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(12,6))
plt.subplot(411)
plt.plot(log_returns, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
#Now we shall examine the serial correlation
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data.log_return, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data.log_return, lags=40, ax=ax2)
plt.show()
#Let's try to make an Auto Regressive Moving Average model
#I have found this code to find best paramaters for Ar(p) and Ma(q)
from statsmodels.tsa.stattools import ARMA
def best_AR_MA_checker(df,lower,upper):
    from statsmodels.tsa.stattools import ARMA
    from statsmodels.tsa.stattools import adfuller
    arg=np.arange(lower,upper)
    arg1=np.arange(lower,upper)
    best_param_i=0
    best_param_j=0
    temp=12000000
    rs=99
    for i in arg:
        for j in arg1:
            model=ARMA(df, order=(i,0,j))
            result=model.fit(disp=0)
            resid=adfuller(result.resid)
            if (result.aic<temp and  adfuller(result.resid)[1]<0.05):
                temp=result.aic
                best_param_i=i
                best_param_j=j
                rs=resid[1]
                
                
            print ("AR: %d, MA: %d, AIC: %d; resid stationarity check: %d"%(i,j,result.aic,resid[1]))
            
    print("the following function prints AIC criteria and finds the paramters for minimum AIC criteria")        
    print("best AR: %d, best MA: %d, best AIC: %d;  resid stationarity check:%d"%(best_param_i, best_param_j, temp, rs))     
best_AR_MA_checker(data.log_return,0,3) #For each parameter I want to try from 0 to 2
#So, I wanna try arma(1,0)
from statsmodels.tsa.stattools import ARMA
model=ARMA(data.log_return, order=(1,0))
res=model.fit(disp=0)
print (res.summary())
