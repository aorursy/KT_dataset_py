import pandas as pd
from pandas.plotting import lag_plot
import numpy as np
import sklearn as sk
from sklearn import preprocessing as pr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import correlate
from scipy.stats.mstats import spearmanr
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.preprocessing import PolynomialFeatures


# create arrays of fake points
x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
# fit up to deg=3
z = np.polyfit(x, y, 3)
#print(z)
# Create matrix and vectors



p = np.poly1d(z)
p30 = np.poly1d(z)
p30(4)
p30(10)
xp = np.linspace(-2, 6, 200)
plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
plt.ylim(-2,2)
print(x.shape, y.shape)
plt.show()
########################################################################################
# THIS CHAPTER IS FOR THE REAL ANALYSIS WITH SELECTED COINS
# BTC, DASH, DOGE, ETH, LTC, XMR
# ITS ABOUT USING ML ALGORITHM 'LINEAR REGRESSION' AND CROSS CORRELATIONS BETWEEN THE COINS
########################################################################################
import pandas as pd
from pandas.plotting import lag_plot
import numpy as np
import sklearn as sk
from sklearn import preprocessing as pr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import correlate
from scipy.stats.mstats import spearmanr
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.graphics.tsaplots import plot_pacf


crypto = {}
crypto['bitcoin'] = pd.read_csv('../input/bitcoin-altcoins-in-2017/CHART_DATA_BITCOIN_2017.csv')
crypto['dash'] = pd.read_csv("../input/cc2017/DASH_2017.csv")
crypto['ethereum'] = pd.read_csv("../input/cc2017/ETH_2017.csv")
crypto['litecoin'] = pd.read_csv("../input/cc2017/LTC_2017.csv")
crypto['monero'] = pd.read_csv("../input/cc2017/XMR_2017.csv")
crypto['dogecoin'] = pd.read_csv("../input/cc2017/DOGE_2017.csv")

# For this analysis I will only be looking at weighted AVerage price to make things more manageable
for coin in crypto:
    for column in crypto[coin].columns:
        if column not in ['date', 'weightedAverage']:
            crypto[coin] = crypto[coin].drop(column, 1)
    # Make date the datetime type and reindex
    crypto[coin] = crypto[coin].set_index(crypto[coin]['date'])
    # delete the double coloumn 'date'
    crypto[coin] = crypto[coin].drop('date', 1)
    
# DEUGGING VALUES OF ALL COINS 
#for coin in crypto:
#    print(crypto[coin], coin, len(crypto[coin]))

# Difference per row (from day to day)
# get difference over rows for the column 'weightedAverage' of all coins
# replace all NaN elements with 0s and 
# store in a newly created column in the dataframe called 'weightedAverageDiff
for coin in crypto:
    crypto[coin]['weightedAverageDiff'] = crypto[coin]['weightedAverage'].diff().fillna(0)

# visualize the calculated differences of weighted average price 
# from day to day of all coins in one plot
for coin in crypto:
    plt.plot(crypto[coin]['weightedAverageDiff'], label=coin)
plt.legend(loc=2)
plt.title('Daily Differenced weightedAverage Prices')
plt.subplots_adjust(left=0.5, right=3.0, top=0.9, bottom=0.1)
plt.xticks(rotation=90)
plt.xticks(np.arange(np.datetime64('2017-01-01'), np.datetime64('2017-12-31'), np.timedelta64(30,'D')))
plt.show()

# Percent Change per row (from day to day)
# get the percent change value for each row for each coin 
# replace all NaN elements with 0s and 
# store in a newly created column in the dataframe called 'weightedAveragePercentChange'
for coin in crypto:
    crypto[coin]['weightedAveragePercentChange'] = crypto[coin]['weightedAverage'].pct_change().fillna(0)
    
for coin in crypto:
    plt.plot(crypto[coin]['weightedAveragePercentChange'], label=coin)
plt.legend(loc=2)
plt.title('Daily Percent Change of weightedAverage Price')
plt.subplots_adjust(left=0.5, right=3.0, top=0.9, bottom=0.1)
plt.xticks(rotation=90)
plt.xticks(np.arange(np.datetime64('2017-01-01'), np.datetime64('2017-12-31'), np.timedelta64(30,'D')))
plt.show()    
################################################################################
# CORRELATION BETWEEN COINS
################################################################################
# As previously stated, the goal of this analysis is to create a correlation matrix 
# for these currencies. One way to find correlation between timeseries is to look at 
# cross-correlation of the timeseries. Cross-correlation is computed between 
# two timeseries using a lag, so when creating the correlation matrix I will 
# specify the correlation as well as the lag.
# Before computing the cross correlation, it is important to have wide-sense station 
# (often just called stationary) data. There are a few ways to make data stationary-- 
# one of which is through differencing. But even after this it is famously difficult 
# to avoid spurious correlations between timeseries data that are often caused by autocorrelation. 
# See this article for an in depth analysis of how spurious correlations arise and how to avoid them: 
# https://link.springer.com/article/10.3758/s13428-015-0611-2.
# For now I employ daily differencing (as it is not seasonal) and test for stationarity to 
# prepare for cross correlation testing.    
corrBitcoin = {}
corrDF = pd.DataFrame()

for coin in crypto: 
    corrBitcoin[coin] = correlate(crypto[coin]['weightedAveragePercentChange'], crypto['bitcoin']['weightedAveragePercentChange'])
    lag = np.argmax(corrBitcoin[coin])
    laggedCoin = np.roll(crypto[coin]['weightedAveragePercentChange'], shift=int(np.ceil(lag)))
    corrDF[coin] = laggedCoin
    plt.figure(figsize=(15,10))
    plt.plot(laggedCoin)
    plt.plot(crypto['bitcoin']['weightedAveragePercentChange'].values)
    title = coin + '/bitcoin PctChg lag: ' + str(lag-349)
    plt.subplots_adjust(left=0.5, right=3.0, top=0.9, bottom=0.1)
    plt.title(title)
    plt.show()

    
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
}

plt.matshow(corrDF.corr(method='pearson'))
plt.xticks(range(10), corrDF.columns.values, rotation='vertical')
plt.yticks(range(10), corrDF.columns.values)
plt.xlabel('Pearson Correlation', fontdict=font)
plt.show()
corrDF.corr(method='pearson')

plt.matshow(corrDF.corr(method='kendall'))
plt.xticks(range(10), corrDF.columns.values, rotation='vertical')
plt.yticks(range(10), corrDF.columns.values)
plt.xlabel('Kendall Correlation', fontdict = font)
plt.show()
corrDF.corr(method='kendall')

#Regression - Using time-based features such as week, month, day, day of week, etc as predictors. 
# You can also add in external predictors that may influence the target
for coin in crypto:
    model = LinearRegression()
    model.fit(np.arange(365).reshape(-1,1), crypto[coin]['weightedAverage'].values)
    trend = model.predict(np.arange(365).reshape(-1,1))
    regressionPlot = plt.subplot(1, 2, 1)
    plt.plot(trend, label='trend')
    plt.plot(crypto[coin]['weightedAverage'].values)
    plt.title('trend in 2017 of ' + coin + ' with linear regression prediction')
    regressionPlot.set_xlabel('days of 2017')
    regressionPlot.set_ylabel('weighted average price')
    
    normalPlot = plt.subplot(1, 2, 2)
    plt.plot(crypto[coin]['weightedAverage'].values - trend, label='residuals')
    plt.title('trend in 2017 of ' + coin)
    normalPlot.set_xlabel('days of 2017')
    normalPlot.set_ylabel('weighted average price')
   # plt.tight_layout()
    plt.subplots_adjust(left=0.5, right=3.0, top=0.9, bottom=0.1)
    plt.show()


import matplotlib.pylab as plt
import numpy as np
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn import datasets

diabetes = datasets.load_diabetes() # load data
print(diabetes)
print(diabetes.data, diabetes.data.shape)
diabetes.data.shape #

# Sperate train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)
print(y_train)
print(y_test)
# There are three steps to model something with sklearn
# 1. Set up the model
model = LinearRegression()
# 2. Use fit / fits a linear model
model.fit(X_train, y_train)
# 3. Check the score
model.score(X_test, y_test)

model.coef_ # Get the coefficients, beta
model.intercept_ # Get the intercept, c
model.predict(X_test) # Predict unkown data, Predict Y using the linear model with estimated coefficients
# plot prediction and actual data
y_pred = model.predict(X_test) 
plt.plot(y_test, y_pred, '.')
# plot a line, a perfit predict would all fall on this line
x = np.linspace(0, 330, 100)
y = x
plt.plot(x, y)
plt.show()
# Quelle:
# https://www.kaggle.com/andyxie/beginner-scikit-learn-linear-regression-tutorial/notebook
########################################################################################
# THIS CHAPTER IS FOR THE REAL ANALYSIS WITH BTC FROM 2014 to 2017 only
# ITS ABOUT USING ML ALGORITHM 'LINEAR REGRESSION' AND CROSS CORRELATIONS BETWEEN THE 
# DIFFERENT YEAR TRENDS
########################################################################################
import pandas as pd
from pandas.plotting import lag_plot
import numpy as np
import sklearn as sk
from sklearn import preprocessing as pr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from scipy.signal import correlate
from scipy.stats.mstats import spearmanr
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.graphics.tsaplots import plot_pacf
import calendar

crypto = {}
crypto['btc14'] = pd.read_csv('../input/btc-ohlc-history-2014-to-2017/BTC_2014.csv') #2014 365 d
crypto['btc15'] = pd.read_csv('../input/btc-ohlc-history-2014-to-2017/BTC_2015.csv') #2015 365 d
crypto['btc16'] = pd.read_csv('../input/btc-ohlc-history-2014-to-2017/BTC_2016.csv') #2016 Schaltjahr +1 Tag
crypto['btc17'] = pd.read_csv('../input/btc-ohlc-history-2014-to-2017/BTC_2017.csv') #2017 365 d
start = pd.to_datetime(crypto['btc14']['Date'][0])
end = pd.to_datetime(crypto['btc14']['Date'][364])
#print(crypto['btc14']['Close'].values)
for coin in crypto:
    for column in crypto[coin].columns:
        if column not in ['Date', 'Close']:
            crypto[coin] = crypto[coin].drop(column, 1)
    # Make date the datetime type and reindex
    crypto[coin]['Date'] = pd.to_datetime(crypto[coin]['Date'])
    crypto[coin] = crypto[coin].sort_values('Date')
    #crypto[coin] = crypto[coin].set_index(crypto[coin]['Date'])
    # delete the double coloumn 'date'
    #crypto[coin] = crypto[coin].drop('Date', 1)

#print(crypto['btc14']) 

crypto['btc14']['month'] = crypto['btc14']['Date'].dt.strftime('%b')
crypto['btc15']['month'] = crypto['btc15']['Date'].dt.strftime('%b')
crypto['btc16']['month'] = crypto['btc16']['Date'].dt.strftime('%b')
crypto['btc17']['month'] = crypto['btc17']['Date'].dt.strftime('%b')

fig, ax = plt.subplots()
ax.plot(crypto['btc14']['Close'].values)
ax.plot(crypto['btc15']['Close'].values)
ax.plot(crypto['btc16']['Close'].values)
ax.plot(crypto['btc17']['Close'].values/10)
plt.show()
########################################################## VALUES OF ALL COINS 
#for coin in crypto:
#    print(crypto[coin])

#Regression - Using time-based features such as week, month, day, day of week, etc as predictors. 
# You can also add in external predictors that may influence the target
for coin in crypto:
    crypto[coin]['closingDiff'] = crypto[coin]['Close'].diff().fillna(0)

for coin in crypto:
    if coin == 'btc17':
        # reduced the value of bitcoin prices of 2017 to make it more readable
        plt.plot(crypto[coin]['Date'], crypto[coin]['closingDiff']/20, label=coin)
    else:
        plt.plot(crypto[coin]['Date'], crypto[coin]['closingDiff'], label=coin)
plt.legend(loc=2)
plt.title('Daily Differenced Closing Prices')
plt.subplots_adjust(left=0.5, right=3.0, top=0.9, bottom=0.1)
plt.xticks(rotation=90)
plt.xticks(np.arange(np.datetime64('2014-01-01'), np.datetime64('2017-12-31'), np.timedelta64(30,'D')))
plt.show()

# Percent Change
for coin in crypto:
    crypto[coin]['ClosePercentChange'] = crypto[coin]['Close'].pct_change().fillna(0)
    
for coin in crypto:
    plt.plot(crypto[coin]['Date'], crypto[coin]['ClosePercentChange'], label=coin)
plt.legend(loc=2)
plt.title('Daily Percent Change of Closing Price')
plt.subplots_adjust(left=0.5, right=3.0, top=0.9, bottom=0.1)
plt.xticks(rotation=90)
plt.xticks(np.arange(np.datetime64('2014-01-01'), np.datetime64('2017-12-31'), np.timedelta64(30,'D')))
plt.grid()
plt.show()    

##########################################################################################
# daily BTC change trends overlapped
##########################################################################################
ax = plt.subplot()    
ax.plot(crypto['btc14']['Date'].values, crypto['btc14']['ClosePercentChange'].values)
ax.plot(crypto['btc14']['Date'].values, crypto['btc15']['ClosePercentChange'].values)
ax.plot(crypto['btc14']['Date'].values, crypto['btc16']['ClosePercentChange'].values[0:365])
ax.plot(crypto['btc14']['Date'].values, crypto['btc17']['ClosePercentChange'].values)
plt.title('daily BTC change trends overlapped')

#plt.xticks(np.arange(np.datetime64('01-01'), np.datetime64('31-12'), np.timedelta64(1,'M')))
#plt.xticks(np.arange(np.datetime64('2014-01-01'), np.datetime64('2017-12-31'), np.timedelta64(50,'D')))
#ax.set_xticks(np.arange(48), calendar.month_name[1:13])
#plt.xticks(np.arange(np.datetime64(''), np.datetime64('2017-12-31'), np.timedelta64(1,'D')))
plt.subplots_adjust(right=5.0, top=0.9, bottom=0.1)
ax.set_xlim([min(crypto['btc14']['Date'].values), max(crypto['btc14']['Date'].values)])
months = mdates.MonthLocator()
monthsFmt = mdates.DateFormatter('%b')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_xlabel('months')
ax.set_ylabel('closing price in %')
plt.show() 

##############################################################################
# using simple ML ALGORITHM LINEAR REGRESSION ON EACH BTC TREND 2014-2017
##############################################################################

for coin in crypto:
    model = LinearRegression()
    if coin == 'btc16':
        model.fit(np.arange(366).reshape(-1,1), crypto[coin]['Close'].values)
        trend = model.predict(np.arange(366).reshape(-1,1))
    else:
        model.fit(np.arange(365).reshape(-1,1), crypto[coin]['Close'].values)
        trend = model.predict(np.arange(365).reshape(-1,1))
    regressionPlot = plt.subplot(1, 2, 1)
    plt.plot(trend, label='trend')
    plt.plot(crypto[coin]['Close'].values)
    plt.title('trend in 2017 of ' + coin + ' with linear regression prediction')
    regressionPlot.set_xlabel('days of 2017')
    regressionPlot.set_ylabel('Closing price')
    
    normalPlot = plt.subplot(1, 2, 2)
    plt.plot(crypto[coin]['Close'].values - trend, label='residuals')
    plt.title('trend in 2017 of ' + coin)
    normalPlot.set_xlabel('days of 2017')
    normalPlot.set_ylabel('Closee price')
   # plt.tight_layout()
    plt.subplots_adjust(left=0.5, right=3.0, top=0.9, bottom=0.1)
    plt.show()
##############################################################################
# using simple ML ALGORITHM LINEAR REGRESSION ON BTC TREND 2014
# SPLITTING DATA RANDOMLY 
#In practice you wont implement linear regression on the entire data set, 
# you will have to split the data sets into training and test data sets. 
# So that you train your model on training data and see how well it performed
# on test data.
##############################################################################
# You have to divide your data sets randomly. 
# Scikit learn provides a function called train_test_split to do this.

##### EVENTUELL DEFAULT WERTE ALS REFERENZ NEHMEN WENN REGRESSION NICHT GUT WIRD
#### WENN ES DANN AUCH KEINEN GUTEN SCORE ZURÜCK GIBT, DANN IST ES ZU VOLATILE, KEIN GUTES MODEL
for column in crypto['btc14'].columns:
        if column not in ['Close', 'closingDiff', 'ClosePercentChange']:
            crypto['btc14'] = crypto['btc14'].drop(column, 1)
#print(crypto['btc14'].shape) #(365,5)
btc = pd.DataFrame(crypto['btc14'])
#btc.head()
#   drop the price column ('Close) as I want only the parameters as my X values
X = btc.drop('Close', axis = 1)
# debug 
#print(btc)
lm = LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, btc.Close, test_size=0.33, random_state=5)
#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)
lm.fit(X_train, Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)
print("Fit a model X_train, and calculate MSE with Y_train: ") 
print(np.mean((Y_train - pred_train) ** 2))
print("Fit a model X_train, and calculate MSE with X_test, Y_test: ")
print(np.mean((Y_test - pred_test) ** 2))

plt.scatter(lm.predict(X_train), lm.predict(X_train) - Y_train, c='g', s=10, alpha=0.2)
plt.scatter(lm.predict(X_test), lm.predict(X_test) - Y_test, c='b', s=10)
plt.hlines(y = 0, xmin= 300, xmax = 600)
plt.title('Residual plot mit training und test data')
plt.ylabel('resiuals')
btc = pd.read_csv('../input/btc-ohlc-history-2014-to-2017/BTC_2014.csv') #2014 365 d
for column in btc.columns:
        if column not in ['Open', 'High', 'Low', 'Close']:
            btc = btc.drop(column, 1)
#print(btc.shape) #(365,5)
btc = pd.DataFrame(btc)
#btc.head()
# drop the price column ('Close) as I want only the parameters as my X values
X = btc.drop('Close', axis = 1)
lm = LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, btc.Close, test_size=0.33, random_state=5)
#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)
lm.fit(X_train, Y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)
print("Fit a model X_train, and calculate MSE with Y_train: ") 
print(np.mean((Y_train - pred_train) ** 2))
print("Fit a model X_train, and calculate MSE with X_test, Y_test: ")
print(np.mean((Y_test - pred_test) ** 2))
########################################################################################
# MITTLERE QUADRATISCHE AWEICHUNG (MQE)
# Eine geringe mittlere quadratische Abweichung bedeutet im klassischen Fall, 
# dass gleichzeitig Bias und Varianz des Schätzers klein sind. 
# Man befindet sich mit dem Schätzer also im Mittel in der Nähe des zu 
# schätzenden Funktionals (geringer Bias) und weiß gleichzeitig, dass die Schätzwerte 
# wenig streuen (geringe Varianz) und mit großer Wahrscheinlichkeit auch in der Nähe ihres 
# Erwartungswerts liegen.
########################################################################################
########################################################################################
# A residual value is a measure of how much a regression line vertically misses a data point. 
# Regression lines are the best fit of a set of data. You can think of the lines as averages; 
# a few data points will fit the line and others will miss. 
# A residual plot has the Residual Values on the vertical axis; the horizontal axis displays 
# the independent variable.
########################################################################################

plt.scatter(lm.predict(X_train), lm.predict(X_train) - Y_train, c='g', s=10, alpha=0.2)
plt.scatter(lm.predict(X_test), lm.predict(X_test) - Y_test, c='b', s=10)
plt.hlines(y = 0, xmin= 300, xmax = 600)
plt.title('Residual plot mit training und test data')
plt.ylabel('resiuals')

