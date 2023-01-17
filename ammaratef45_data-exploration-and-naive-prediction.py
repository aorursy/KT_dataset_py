# import libraries to be used
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
import datetime as datetime
import matplotlib.ticker as ticker
from matplotlib.dates import num2date, date2num
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

%matplotlib inline

# discover inputs
print(os.listdir("../input/Data"))
# Stocks directory has data for stocks, every stock in a file
# ُETFs directory has data for ETFs, every ETF in a file

print("Reading data from " + os.listdir("../input/Data/Stocks")[0] + "file in stocks")
stock = pd.read_csv("../input/Data/Stocks/" + os.listdir("../input/Data/Stocks")[0])
print("shape: ({},{})".format(stock.shape[0], stock.shape[1]))
display(stock.head(n=2))

print("Reading data from " + os.listdir("../input/Data/ETFs")[0] + "file in stocks")
etf = pd.read_csv("../input/Data/ETFs/" + os.listdir("../input/Data/ETFs")[0])
print("shape: ({},{})".format(stock.shape[0], stock.shape[1]))
display(etf.head(n=2))
def readAllDataframes():
    dataframes = []
    nonreadable = 0
    for f in os.listdir("../input/Data/Stocks"):
        try:
            dataframes.append(pd.read_csv("../input/Data/Stocks/" + f))
        except:
            nonreadable += 1
    for f in os.listdir("../input/Data/ETFs"):
        try:
            dataframes.append(pd.read_csv("../input/Data/ETFs/" + f))
        except:
            nonreadable += 1
    return dataframes,nonreadable

allDataFrames, nonparsable = readAllDataframes()
print("Couldn't read total of {} files".format(nonparsable))

def rowNumbers(dataframes):
    numbers = []
    for d in dataframes:
        numbers.append(d.shape[0])
    return numbers
n = rowNumbers(allDataFrames)
print("mean is: {}".format(np.mean(n)))
print("median is: {}".format(np.median(n)))
print("maximum is: {}".format(np.max(n)))
print("minimum is: {}".format(np.min(n)))
size = len(allDataFrames)
print("current size is: {}".format(size))
indToRemove = []
for i in range(size):
    if allDataFrames[i].shape[0]<1000:
        indToRemove.append(i)
allDataFrames = [i for j, i in enumerate(allDataFrames) if j not in indToRemove]
print("size after removing is: {}".format(len(allDataFrames)))
print("removed total of {} data frames".format(size-len(allDataFrames)))
def drawCandles(dataframe, size):
    dateconv = lambda s: datetime.datetime.strptime(s, '%Y-%m-%d')
    quotes = [tuple([date2num(dateconv(dataframe['Date'][:size][i])),
                 dataframe['Open'][:size][i],
                 dataframe['High'][:size][i],
                 dataframe['Low'][:size][i],
                 dataframe['Close'][:size][i]]) for i in range(size)]
    fig, ax = plt.subplots()
    candlestick_ohlc(ax, quotes, width=0.2, colorup='r', colordown='k', alpha=1.0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()
def drawSeries(dataframe):
    plt.plot(dataframe['Date'], dataframe['Close'])
    plt.show()
def mydate(x,pos):
    return "{}-{}-{}".format(num2date(x).year, num2date(x).month, num2date(x).day)
# Candle Sticks are famous among stock traders, this is going to draw first 20 values in candle sticks format
drawCandles(allDataFrames[0], 20)
# the chart for close price for all provided data
drawSeries(allDataFrames[0])
dateconv = lambda s: datetime.datetime.strptime(s, '%Y-%m-%d')
frame = allDataFrames[0]
X = []
for i in range(len(frame['Date'])):
    X.append(date2num(dateconv(frame['Date'][i])))
X = np.array(X)
X = X.reshape(-1, 1)
Y = []
for i in range(len(frame['Close'])):
    Y.append(frame['Close'].to_frame().to_records(index=False)[i][0])
Y = np.array(Y)
Y = Y.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42, shuffle=False)
model = LinearRegression()
model.fit(X_train, Y_train)
prediction = model.predict(X_test)
print("Score: {}".format(model.score(prediction, Y_test)))
plt.plot(X_train, Y_train, color='g')
plt.plot(X_train, model.predict(X_train), color='k')
plt.plot(X_test, Y_test, color='g')
plt.plot(X_test, prediction,color='r')
plt.show()