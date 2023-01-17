import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from scipy import stats
nifty=pd.read_csv("../input//Nifty_data.csv")
nifty.head()
nifty_returns=(nifty['Close']/nifty['Open'])-1

volatility= np.std(nifty_returns)
trading_days=len(nifty_returns)
mean=(nifty.loc[trading_days-1,'Close']/nifty.loc[0,'Open'])-1

print('Annual Average Nifty return',mean)
print('Annual volatility',volatility*np.sqrt(trading_days))
print('Number of trading days',trading_days)
daily_returns=np.random.normal(mean/trading_days,volatility,trading_days)+1

index_returns=[10980]  
                               
for x in daily_returns:
    index_returns.append(index_returns[-1]*x)

plt.plot(index_returns)
plt.show()
for i in range(1000):
    daily_returns=np.random.normal(mean/trading_days,volatility,trading_days)+1

    index_returns=[10980]  
    
    for x in daily_returns:
        index_returns.append(index_returns[-1]*x)

    plt.plot(index_returns)

plt.show()
index_result=[]

for i in range(1000):
    daily_returns=np.random.normal(mean/trading_days,volatility,trading_days)+1

    index_returns=[10980]  
    
    for x in daily_returns:
        index_returns.append(index_returns[-1]*x)
 
    index_result.append(index_returns[-1])

plt.hist(index_result)
plt.show()
print('Average expected value of Nifty:',np.mean(index_result))
print('10 percentile:',np.percentile(index_result,10))
print('90 percentile:',np.percentile(index_result,90))
from fbprophet import Prophet
nifty.head()
nifty=nifty.iloc[:,0:2]
nifty.head()
nifty['Date']= pd.to_datetime(nifty['Date'])
nifty.rename(columns={'Date':'ds','Open':'y'},inplace=True)
model=Prophet()
model.fit(nifty)

predict_df=model.make_future_dataframe(periods=252)
predict_df.tail()
forecast=model.predict(predict_df)
forecast.tail()
fig1=model.plot(forecast)
fig2=model.plot_components(forecast)