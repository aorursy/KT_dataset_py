# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
# Define the instruments to download. We would like to see Apple, Microsoft and the S&P500 index.
tickers = ['AAPL'] #, 'MSFT', 'IBM'] (you can add more tickers as a list)

# We would like all available data from 01/01/2017 until 12/31/2017.
start_date = '2017-01-01' # you can set this to whatever date you want
end_date = dt.datetime.now() # this puts the current time into a variable called end_date

# This next function creates a pandas dataframe containing the results of the DataReader query
# The 'yahoo' datasource provides the stock ticker info. (google and morningstar no longer work).
# The results are stored as a dataframe called df (nice and short!)
df = web.DataReader(tickers, data_source='yahoo', start=start_date, end=end_date)
# Inspect the first 5 rows
df.head()
df.plot(y='Low') 
df[["High", "Low"]].plot()

plt.style.available
plt.style.use("fivethirtyeight") #need to reset this every time you want to change the template
df[["High", "Low"]].plot()
plt.style.use("ggplot")
df[["High", "Low"]].plot()
google = web.DataReader('GOOG', data_source='yahoo', start=start_date, end=end_date)

google['Close'].mean()
google
def rank_performance(stock_price):
    if stock_price <= 900:
        return "Poor"
    elif stock_price>900 and stock_price <=1200:
        return "Good"
    elif stock_price>1200:
        return "Stellar"
google['Close'].apply(rank_performance)
google
google['Close'].apply(rank_performance).value_counts().plot(kind="bar")
google["Close"].apply(rank_performance).value_counts().plot(kind="barh")
jnj = web.DataReader('JNJ', data_source='yahoo', start='2016-01-01', end=dt.datetime.now())
jnj.head()
jnj['Close'].mean()
def above_or_below(stock_price):
    if stock_price >= 128.33:
        return "Above average"
    else:
        return "Below average"
labels='above','below'
colors = ['mediumseagreen','lightcoral'] 
jnj["Close"].apply(above_or_below).value_counts().plot(kind='pie', legend=False, labels=labels, colors=colors)
cars = pd.read_csv("../input/original-cars/original cars.csv")
cars # show the head and tail of this file
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
size = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=size, c=colors, alpha=0.5)
plt.show()
x=cars[['MPG']]
y=cars[['Horsepower']]
cars[['MPG','Horsepower']].plot(kind='scatter', x='MPG', y='Horsepower', alpha=0.5)
cars[['Acceleration','Horsepower']].plot(kind='scatter',x='Acceleration', y='Horsepower',  alpha=0.5, legend=True)
size=cars[['Displacement']]
cars[['MPG','Horsepower']].plot(kind='scatter', x='MPG', y='Horsepower', alpha=0.5, legend=True, s=size, figsize=(12,8))
hist=cars.hist(column='MPG')
hist=cars.hist(figsize=(12,8))
hist = cars.hist(column='MPG', bins=10, grid=False, figsize=(12,8), color='#4290be', zorder=2, rwidth=0.9)

hist = hist[0] # each unique value is accessed by its index (the car name) which is in clumn 0

for x in hist:

    # Switch off tickmarks
    x.tick_params(axis="both", which="both", bottom=False, top=True, labelbottom=True, left=False, right=False, labelleft=True)

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Set title (set to "" for no title!)
    x.set_title("Cars and MPG")

    # Set x-axis label
    x.set_xlabel("Miles per Gallon", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("Number of cars", labelpad=20, weight='bold', size=12)