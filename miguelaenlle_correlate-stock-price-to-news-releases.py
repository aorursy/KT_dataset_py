!pip install yfinance # for getting stock data

import yfinance as yf

import pandas as pd # load in the data

import matplotlib.pyplot as plt # for plotting
reuters_data = pd.read_csv('/kaggle/input/reuters_data.csv', index_col = 0)
stock = 'AA' # example stock

reuters_stock_data = reuters_data[reuters_data['stock'] == stock] # get all news releases for AA stock
articles_with_sentiment = reuters_stock_data[reuters_stock_data['compound_sentiment'].abs() > 0]

# get all stocks with compound sentiments over that aren't zero
articles_with_sentiment['article_publish_date'] = pd.DatetimeIndex(articles_with_sentiment['article_publish_date'])
# get stock price data

stock_data = yf.download(stock, 

                         (articles_with_sentiment['article_publish_date'].min() - pd.Timedelta('5 days')).date(), 

                         (articles_with_sentiment['article_publish_date'].max() + pd.Timedelta('5 days')).date())
maximum_alpha_value = 1.0

for i in articles_with_sentiment.index:

    # get sentiment

    sentiment = articles_with_sentiment.loc[i, 'compound_sentiment']

    if sentiment <= 0:

        color = 'red' # the color of the vertical line will be red if sentiment < 0

    else:

        color = 'green'# the color of the vertical line will be green if sentiment > 0

    # alpha (transparency) of the line is dependent on how strong the

    # article's sentiment is

    plt.axvline(articles_with_sentiment.loc[i, 'article_publish_date'], color = color, alpha = maximum_alpha_value * abs(sentiment))

plt.plot(stock_data['Open'])

plt.show()