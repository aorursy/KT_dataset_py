%matplotlib inline



"""Compute daily returns."""



import os

import datetime

import pandas as pd

import matplotlib.pyplot as plt



def symbol_to_path(symbol, base_dir="../input/CryptoCurrencyHistoricalData/"):

    """Return CSV file path given ticker symbol."""

    return os.path.join(base_dir, "{}.csv".format(str(symbol)))





def fix_df_date(data):

    # Credits to https://www.kaggle.com/josephmiguel/loading-fixing-and-plotting-data/notebook

    bad_data = data[data.Date.map(lambda x: x.find('v')>0)]

    bad_data['Date'] = bad_data['Date'].map(lambda x: x.replace('v', '/'))

    fixed_data = data.copy()

    fixed_data['Date'] = fixed_data['Date'].map(lambda x: x.replace('v', '/'))

    fixed_data['Date'] = [datetime.datetime.strptime(_, '%d/%m/%Y') for _ in fixed_data['Date']]

    fixed_data = fixed_data.sort_values(by='Date')

    fixed_data.set_index('Date', inplace=True)

    return fixed_data

    

    

def get_data(symbols, dates):

    """Read stock data (adjusted close) for given symbols from CSV files."""

    df = pd.DataFrame(index=dates)

    

    for symbol in symbols:

        df_temp = pd.read_csv(symbol_to_path(symbol), delimiter=';',

            parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])

        df_temp = df_temp.rename(columns={'Close': symbol})

        df_temp = fix_df_date(df_temp)

        df = df.join(df_temp)

        

    df = df.dropna()

    return df





def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):

    """Plot stock prices with a custom title and meaningful axis labels."""

    ax = df.plot(title=title, fontsize=12, figsize=(20,10))

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    plt.show()





def compute_daily_returns(df):

    """Compute and return the daily return values."""

    daily_returns = (df/df.shift(1)) - 1

    daily_returns.ix[0, :] = 0

    return daily_returns





def test_run():

    # Read data

    dates = pd.date_range('2017-01-01', '2017-12-31')

    # symbols = ['bitcoin','litecoin','iota','dash','ripple']

    symbols = ['bitcoin','dash']

    df = get_data(symbols, dates)

    # plot_data(df)



    # Compute daily returns

    daily_returns = compute_daily_returns(df)

    plot_data(daily_returns, title="Daily returns", ylabel="Daily returns")



    # Compute daily returns



if __name__ == "__main__":

    test_run()
