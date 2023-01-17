# We import pandas into Python
import pandas as pd

# We read in a stock data data file into a data frame and see what it looks like
df = pd.read_csv('../input/GOOG.csv')

# We display the first 5 rows of the DataFrame
df.head()
# We load the Google stock data into a DataFrame
google_stock = pd.read_csv('../input/GOOG.csv', index_col=['Date'], usecols=['Date', 'Adj Close'], parse_dates=True)

# We load the Apple stock data into a DataFrame
apple_stock = pd.read_csv('../input/AAPL.csv', index_col=['Date'], usecols=['Date', 'Adj Close'], parse_dates=True)

# We load the Amazon stock data into a DataFrame
amazon_stock = pd.read_csv('../input/AMZN.csv', index_col=['Date'], usecols=['Date', 'Adj Close'], parse_dates=True)
# We display the google_stock DataFrame
apple_stock.head()
# We create calendar dates between '2000-01-01' and  '2016-12-31'
dates = pd.date_range('2000-01-01', '2016-12-31')

# We create and empty DataFrame that uses the above dates as indices
all_stocks = pd.DataFrame(index = dates)
# Change the Adj Close column label to Google
google_stock = google_stock.rename(index=str, columns={'Adj Close':'google_stock'})

# Change the Adj Close column label to Apple
apple_stock = apple_stock.rename(index=str, columns={'Adj Close':'apple_stock'})

# Change the Adj Close column label to Amazon
amazon_stock = amazon_stock.rename(index=str, columns={'Adj Close':'amazon_stock'})
# We display the google_stock DataFrame
google_stock.head()
# We join the Google stock to all_stocks
all_stocks = all_stocks.join(google_stock)

# We join the Apple stock to all_stocks
all_stocks = all_stocks.join(apple_stock)

# We join the Amazon stock to all_stocks
all_stocks = all_stocks.join(amazon_stock)
# We display the google_stock DataFrame
all_stocks.head()
# Check if there are any NaN values in the all_stocks dataframe
all_stocks.isnull().any()

# Remove any rows that contain NaN values
all_stocks.dropna(axis=0)
# Print the average stock price for each stock
print('Average Stock Price:\n', all_stocks.mean())
# Print the median stock price for each stock
print('\nMedian Stock Price:\n', all_stocks.median())
# Print the standard deviation of the stock price for each stock  
print('\nStandard Deviation:\n', all_stocks.std())
# Print the correlation between stocks
print('\nCorrelation Beween Stocks:\n', all_stocks.corr())
# We compute the rolling mean using a 150-Day window for Google stock
rollingMean = all_stocks['google_stock'].rolling(150).mean()
rollingMean
# this allows plots to be rendered in the notebook
%matplotlib inline 

# We import matplotlib into Python
import matplotlib.pyplot as plt


# We plot the Google stock data
plt.plot(all_stocks['google_stock'])

# We plot the rolling mean ontop of our Google stock data
plt.plot(rollingMean)
plt.legend(['Google Stock Price', 'Rolling Mean'])
plt.show()
