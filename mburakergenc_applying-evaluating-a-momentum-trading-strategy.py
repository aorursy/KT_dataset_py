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
import pandas as pd

import numpy as np

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

from scipy import stats
#################### Helper Stylers ##########################



color_scheme = {

    'index': '#B6B2CF',

    'etf': '#2D3ECF',

    'tracking_error': '#6F91DE',

    'df_header': 'silver',

    'df_value': 'white',

    'df_line': 'silver',

    'heatmap_colorscale': [(0, '#6F91DE'), (0.5, 'grey'), (1, 'red')],

    'background_label': '#9dbdd5',

    'low_value': '#B6B2CF',

    'high_value': '#2D3ECF',

    'y_axis_2_text_color': 'grey',

    'shadow': 'rgba(0, 0, 0, 0.75)',

    'major_line': '#2D3ECF',

    'minor_line': '#B6B2CF',

    'main_line': 'black'}



def generate_config():

    return {'showLink': False, 'displayModeBar': False, 'showAxisRangeEntryBoxes': True}

#################### Helper Methods ##########################

import helper

import plotly.graph_objs as go

import plotly.offline as offline_py

offline_py.init_notebook_mode(connected=True)





def _generate_stock_trace(prices):

    return go.Scatter(

        name='Index',

        x=prices.index,

        y=prices,

        line={'color': color_scheme['major_line']})





def _generate_traces(name_df_color_data):

    traces = []



    for name, df, color in name_df_color_data:

        traces.append(go.Scatter(

            name=name,

            x=df.index,

            y=df,

            mode='lines',

            line={'color': color}))



    return traces





def plot_stock(prices, title):

    config = generate_config()

    layout = go.Layout(title=title)



    stock_trace = _generate_stock_trace(prices)



    offline_py.iplot({'data': [stock_trace], 'layout': layout}, config=config)





def print_dataframe(df, n_rows=10, n_columns=3):

    missing_val_str = '...'

    config = generate_config()



    formatted_df = df.iloc[:n_rows, :n_columns]

    formatted_df = formatted_df.applymap('{:.3f}'.format)



    if len(df.columns) > n_columns:

        formatted_df[missing_val_str] = [missing_val_str]*len(formatted_df.index)

    if len(df.index) > n_rows:

        formatted_df.loc[missing_val_str] = [missing_val_str]*len(formatted_df.columns)



    trace = go.Table(

        type='table',

        columnwidth=[1, 3],

        header={

            'values': [''] + list(formatted_df.columns.values),

            'line': {'color': color_scheme['df_line']},

            'fill': {'color': color_scheme['df_header']},

            'font': {'size': 13}},

        cells={

            'values': formatted_df.reset_index().values.T,

            'line': {'color': color_scheme['df_line']},

            'fill': {'color': [color_scheme['df_header'], color_scheme['df_value']]},

            'font': {'size': 13}})



    offline_py.iplot([trace], config=config)





def plot_resampled_prices(df_resampled, df, title):

    config = generate_config()

    layout = go.Layout(title=title)



    traces = _generate_traces([

        ('Monthly Close', df_resampled, color_scheme['major_line']),

        ('Close', df, color_scheme['minor_line'])])



    offline_py.iplot({'data': traces, 'layout': layout}, config=config)





def plot_returns(returns, title):

    config = generate_config()

    layout = go.Layout(title=title)



    traces = _generate_traces([

        ('Returns', returns, color_scheme['major_line'])])



    offline_py.iplot({'data': traces, 'layout': layout}, config=config)





def plot_shifted_returns(df_shited, df, title):

    config = generate_config()

    layout = go.Layout(title=title)



    traces = _generate_traces([

        ('Shifted Returns', df_shited, color_scheme['major_line']),

        ('Returns', df, color_scheme['minor_line'])])



    offline_py.iplot({'data': traces, 'layout': layout}, config=config)





def print_top(df, name, top_n=10):

    print('{} Most {}:'.format(top_n, name))

    print(', '.join(df.sum().sort_values(ascending=False).index[:top_n].values.tolist()))

df = pd.read_csv('../input/cryptocurrency-market-history-coinmarketcap/all_currencies.csv', parse_dates=['Date'], index_col=False)
# Get top 20 cryptos based on market cap. Most of them are "shitcoins" and have low volume

indicies = df.groupby('Symbol').agg({'Market Cap': max}).sort_values(by='Market Cap', ascending=False).index[:20]
# Create a new pivot table to work with the data and fill missing values with latest closing prices. 

close = df.reset_index().pivot(index='Date', columns='Symbol', values='Close')

close.drop(close.columns.difference(indicies), axis=1, inplace=True)

close.fillna(method='bfill', inplace=True)

close.head()
# Throughout the notebook we will mainly visualize Bitcoin

ticker = 'BTC'

plot_stock(close[ticker], '{} Price'.format(ticker))
def resample_prices(close_prices, freq='M'):

    """

    Resamples close prices for each ticker at specified frequency.

    """   

    return close_prices.resample(freq).last()
monthly_close = resample_prices(close)

monthly_close.head()
plot_resampled_prices(

    monthly_close.loc[:, ticker],

    close.loc[:, ticker],

    '{} Close Vs Monthly Close'.format(ticker))
def compute_log_returns(prices):

    """

    Computes the log returns on given prices

    """

    log_returns = np.log(prices) - np.log(prices.shift(1))

    return log_returns
# Calculate monthly returns

monthly_close_returns = compute_log_returns(monthly_close)
plot_returns(

    monthly_close_returns.loc[:, ticker],

    'Log Returns of {} Stock (Monthly)'.format(ticker))
def shift_returns(returns, n):

    """

    Generate shifted returns to get lookahead and previous returns

    """

    return returns.shift(n)
# Previous month's returns

prev_returns = shift_returns(monthly_close_returns, 1)

# Next month's returns

lookahead_returns = shift_returns(monthly_close_returns, -1)
plot_shifted_returns(

    prev_returns.loc[:, ticker],

    monthly_close_returns.loc[:, ticker],

    'Previous Returns of {} Stock'.format(ticker))
plot_shifted_returns(

    lookahead_returns.loc[:, ticker],

    monthly_close_returns.loc[:, ticker],

    'Lookahead Returns of {} Stock'.format(ticker))
def get_top_n(prev_returns, top_n):

    """

    Return top N cryptos for each day

    """

    top_cryptos = prev_returns.mask(prev_returns.rank(axis = 1, method='max', ascending=False) > top_n, 0)

    top_cryptos = top_cryptos.mask(top_cryptos > 0, int(1))

    top_cryptos = top_cryptos.mask(top_cryptos < 0, int(1))

    top_cryptos.fillna(0, inplace=True)

    return top_cryptos.astype(int)
top_n = 3

df_long = get_top_n(prev_returns, top_n)

df_short = get_top_n(-1*prev_returns, top_n)

print_top(df_long, 'Longed Cryptos')

print_top(df_short, 'Shorted Cryptos')
def portfolio_returns(df_long, df_short, lookahead_returns, top_n_coins):

    """

    Computes expected returns for a portfolio. 

    """

    return ((df_long - df_short) * lookahead_returns) / top_n_coins
# Let's see the returns of our portfolio

expected_portfolio_returns = portfolio_returns(df_long, df_short, lookahead_returns, 2*top_n)



plot_returns(expected_portfolio_returns.T.sum(), 'Portfolio Returns')
# Annualized rate of return

expected_portfolio_returns_by_date = expected_portfolio_returns.T.sum().dropna()

portfolio_ret_mean = expected_portfolio_returns_by_date.mean()

portfolio_ret_ste = expected_portfolio_returns_by_date.sem()

portfolio_ret_annual_rate = (np.exp(portfolio_ret_mean * 12) - 1) * 100



print("""

Mean:                       {:.6f}

Standard Error:             {:.6f}

Annualized Rate of Return:  {:.2f}%

""".format(portfolio_ret_mean, portfolio_ret_ste, portfolio_ret_annual_rate))
# Apply T-Test

def analyze_alpha(expected_portfolio_returns_by_date):

    null_hypothesis = 0

    t, p = stats.ttest_1samp(expected_portfolio_returns_by_date.values, null_hypothesis)

    return t, p/2
t_value, p_value = analyze_alpha(expected_portfolio_returns_by_date)

print("""

Alpha analysis:

 t-value:        {:.3f}

 p-value:        {:.6f}

""".format(t_value, p_value))