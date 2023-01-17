import pandas as pd
import numpy as np

# plotting
import plotly.express as px

# for offline plotting
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
def estimate(data, price, k):
    
    prices = data[price].to_numpy(dtype=np.float64)
    log_prices = np.log(prices)
    rets = np.diff(log_prices)
    T = len(rets)
    mu = np.mean(rets)
    var_1 = np.var(rets, ddof=1, dtype=np.float64)
    rets_k = (log_prices - np.roll(log_prices, k))[k:]
    m = k * (T - k + 1) * (1 - k / T)
    var_k = 1/m * np.sum(np.square(rets_k - k * mu))

    # Variance Ratio
    vr = var_k / var_1
    
    # Phi2
    def delta(j):
        res = 0
        for t in range(j+1, T+1):
            t -= 1  # array index is t-1 for t-th element
            res += np.square((rets[t]-mu)*(rets[t-j]-mu))
        return res / ((T-1) * var_1)**2

    phi2 = 0
    for j in range(1, k):
        phi2 += (2*(k-j)/k)**2 * delta(j)

    return vr, (vr - 1) / np.sqrt(phi2), T
def estimate_multiple_k(data, price, time_intervals):
    
    # Estimate different time_intervals.
    for time_interval in time_intervals:

        vr, stat2, T = estimate(data, price, time_interval)
        print('The number of observations : ' + str(T))
        print('Variance Ratio for k = ' + str(time_interval) + ' : ' + str(vr))
        print('Variance Ratio Test Statistic for k = ' + str(time_interval) + ' Heteroscedasticity Assumption : ' + str(stat2))
        print('-------------------------------------------------------------------------------------------------')
us_index = pd.read_csv('../input/datasource/w1_data.csv', na_values=['.'])
us_index['date'] = pd.to_datetime(us_index['Date'], format='%Y%m%d')
us_index.tail()
# Random generator is not truly random, but pseudo-random
# Specifying a random seed allows us to replicate the set of results. Otherwise, results will differ for each run.
# Note that the seed used here is diffeent from that of the bootcamp's
np.random.seed(1)
# Generate random daily returns using US equity market's historical annual return: mean=8% and standard deviation=20%
# size specifies the number of daily returns to generate
steps = np.random.normal(0.08/252, 0.2/np.sqrt(252), size=10000)
print("We have generated " + str(len(steps)) + " returns")
steps
type(steps)
fig = px.scatter(y=steps, labels={'x':'Time', 'y':'Return'})
fig.show()
# Set first element to 1 so that the first price (defined below) will be the starting stock price
steps[0] = 0
steps
starting_stock_price = 100
# Simulate stock prices, P
# cumsum cumulates the incremental return for each step
P = np.cumprod(1+steps)*starting_stock_price
fig = px.scatter(y=P, labels={'x':'Time', 'y':'Price'})
fig.show()
# Create a dataframe
data = pd.DataFrame(P, columns=['Price'])
data
# Testing different values of k (i.e., time intervals)
estimate_multiple_k(data, 'Price', [2, 4, 8, 16, 32, 64])
estimate_multiple_k(us_index, 'eindend', [2, 4, 8, 16])
estimate_multiple_k(us_index, 'vindend', [2, 4, 8, 16])
def estimate_homosce(data, price, k):
    
    prices = data[price].to_numpy(dtype=np.float64)
    log_prices = np.log(prices)
    rets = np.diff(log_prices)
    T = len(rets)
    mu = np.mean(rets)
    var_1 = np.var(rets, ddof=1, dtype=np.float64)
    rets_k = (log_prices - np.roll(log_prices, k))[k:]
    m = k * (T - k + 1) * (1 - k / T)
    var_k = 1/m * np.sum(np.square(rets_k - k * mu))

    # Variance Ratio
    vr = var_k / var_1
    
    # Phi1
    phi1= (2*(2*k-1)*(k-1))/(3*k*T)

    return vr, (vr - 1) / np.sqrt(phi1), T
def estimate_multiple_k_new(data, price, time_intervals):
    
    # Estimate different time_intervals.
    for time_interval in time_intervals:

        vr, stat1, T = estimate_homosce(data, price, time_interval)
        stat2 = estimate(data, price, time_interval)[1]
        print('The number of observations : ' + str(T))
        print('Variance Ratio for k = ' + str(time_interval) + ' : ' + str(vr))
        print('Variance Ratio Test Statistic for k = ' + str(time_interval) + ' Homoscedasticity Assumption : ' + str(stat1))
        print('Variance Ratio Test Statistic for k = ' + str(time_interval) + ' Heteroscedasticity Assumption : ' + str(stat2))
        print('-------------------------------------------------------------------------------------------------')
#import data from yahoo finance

#run below commented line "!pip..." to install Yfinance package
#!pip install yfinance
import yfinance as yf
tickers = ["UCO","GOEX", "ARKW", "SBIO"]
yf_data = yf.download( tickers = tickers, period = "10y", interval = "1wk", group_by = "ticker")
#yf_data = yf_data.fillna(method = 'ffill')

#UCO etf involves crude oil exposure
#GOEX etf involves gold exploration
#ARKW etf involves tech and internet
#SBIO etf invloves healthcare 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
fig, axs = plt.subplots(2,2)
axs[0,0].plot(yf_data["UCO"].dropna()["Close"])
axs[0,0].set_title('UCO')
axs[0,0].xaxis.set_major_locator(mdates.MonthLocator(interval=36))
axs[0,0].xaxis.set_major_formatter(DateFormatter("%Y"))

axs[0,1].plot(yf_data["GOEX"].dropna()["Close"])
axs[0,1].set_title('GOEX')
axs[0,1].xaxis.set_major_locator(mdates.MonthLocator(interval=36))
axs[0,1].xaxis.set_major_formatter(DateFormatter("%Y"))

axs[1,0].plot(yf_data["ARKW"].dropna()["Close"])
axs[1,0].set_title("ARKW")
axs[1,0].xaxis.set_major_locator(mdates.MonthLocator(interval=24))
axs[1,0].xaxis.set_major_formatter(DateFormatter("%Y"))

axs[1,1].plot(yf_data["SBIO"].dropna()["Close"])
axs[1,1].set_title('SBIO')
axs[1,1].xaxis.set_major_locator(mdates.MonthLocator(interval=24))
axs[1,1].xaxis.set_major_formatter(DateFormatter("%Y"))

plt.tight_layout() 
plt.show()
#ETF analysis
count=0
for ticker in tickers:
    etf = yf_data[ticker].dropna()
    dash=((97-len(ticker))//2)*"-"
    print(dash+ticker+dash+(((97-len(ticker))%2)*"-")+"\n")
    estimate_multiple_k_new(etf, 'Close', [2, 4, 8, 16])
    print("\n"+"\n")
