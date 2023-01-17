%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns

# Make the default figures a bit bigger

plt.rcParams['figure.figsize'] = (7,4.5) 

plt.rcParams["figure.dpi"] = 120 



sns.set(style="ticks")

sns.set_context("notebook", font_scale = .7, rc={"grid.linewidth": 5}) #poster

greek_salad = ['#D0D3C5', '#56B1BF', '#08708A', '#D73A31', '#032B2F']

sns.set_palette(greek_salad)



import warnings

warnings.filterwarnings("ignore")



import glob

filelist = glob.glob("../input/*.csv")



# Load datas

AAPL = pd.read_csv('../input/AAPL.csv', sep=',', index_col=0, parse_dates=True)

cols = ['open', 'high', 'low', 'close', 'unadjustedVolume']

AAPL = AAPL[cols].copy()

AAPL.rename(columns={'unadjustedVolume': 'volume'}, inplace=True)



# display(AAPL.info())

# display(AAPL.head(3))

# display(AAPL.tail(3))



df_stocks = pd.DataFrame()



for i in filelist:

    tmp = pd.read_csv(i)

    tmp['symbol'] = i.split('/')[-1].split('.')[0]

    df_stocks = df_stocks.append(tmp)



cols = ['date', 'open', 'high', 'low', 'close', 'unadjustedVolume', 'changePercent', 'symbol']

df_stocks = df_stocks[cols].copy()

df_stocks.rename(columns={'unadjustedVolume': 'volume', 'changePercent':'change'}, inplace=True)

df_stocks['date'] = pd.to_datetime(df_stocks['date'])

df_stocks.head()
# create new dataframe with just closing price for each stock

df = df_stocks.pivot(index='date', columns='symbol', values='close')



# subset data since 2015

df = df[df.index>='2015-01-01']

df.tail()
### Plot the Daily Closing Price and Volume of a Stock



# plt.figure(figsize=(15,7))

# top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)

# bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)

# top.plot(AAPL.index, AAPL['close'], color=greek_salad[2]) 

# bottom.bar(AAPL.index, AAPL['volume'], color=greek_salad[4]) 

 

# # set the labels

# top.axes.get_xaxis().set_visible(False)

# top.set_title('AAPL')

# top.set_ylabel('Closing Price')

# bottom.set_ylabel('Volume');
### Plot a Histogram of the Daily Closing Price



# plt.figure(figsize=(15,7))

# sns.distplot(AAPL['close'].dropna(), bins=50, color=greek_salad[2]);
# simple moving averages

sma5 = AAPL['close'].rolling(5).mean() #5 days

sma60 = AAPL['close'].rolling(60).mean() #100 days

 

AAPL_sma = pd.DataFrame({'AAPL': AAPL['close'], 'SMA 5': sma5, 'SMA 60': sma60})

AAPL_sma.plot(figsize=(10, 3), legend=True, title='AAPL');

df.plot(figsize=(10,4))

plt.ylabel('Price')

plt.legend("");
returnfstart = df.apply(lambda x: x / x[0])

returnfstart.plot(figsize=(10,4)).axhline(1, lw=3, color='black')

plt.ylabel('Return From Start Price')

plt.legend('', '');
df.pct_change().plot(figsize=(10,4))

plt.axhline(0, color='black', lw=3)

plt.ylabel('Daily Percentage Return')

plt.legend('', '');
# Resample df to business months, take last observation as value 

monthly = df.resample('BM').apply(lambda x: x[-1])



# Calculate the monthly percentage change

monthly.pct_change().plot(figsize=(10,4))

plt.axhline(0, color='black', lw=1)

plt.ylabel('Monthly Percentage Return')

plt.legend('', '');
# Resample df to quarters, take the mean as value per quarter

quarter = df.resample('4M').mean()



# Calculate the quarterly percentage change

quarter.pct_change().plot(figsize=(10,4))

plt.axhline(0, color='black', lw=3)

plt.ylabel('Quarterly Percentage Return')

plt.legend('', '');
# sns.jointplot('AAPL', 'BA', df, kind='reg', color=greek_salad[4]);
# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(7, 7))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5});
# ax1 = sns.PairGrid(df[['AAPL', 'AXP', 'BA']].dropna())

# # define top, bottom and diagonal plots

# ax1.map_upper(plt.scatter, color=greek_salad[4])

# ax1.map_lower(sns.kdeplot, cmap='cool_d')

# ax1.map_diag(sns.distplot, bins=30);
### Distribution of daily percentage change



# daily_pct_c = df/ df.shift(1) -1

# daily_pct_c.hist(bins=50, figsize=(15,20), color=greek_salad[1])

# plt.show()



# display(daily_pct_c.describe())
cum_daily_return = (1 + daily_pct_c).cumprod()



cum_daily_return.plot(figsize=(10,4))

plt.ylabel('Cumulative Daily Returns');

plt.axhline(1, color='black', lw=3)

plt.legend('', '');
# Resample the cumulative daily return to cumulative monthly return 

cum_monthly_return = cum_daily_return.resample("M").mean()



cum_monthly_return.plot(figsize=(10,4))

plt.ylabel('Cumulative Monthly Returns');

plt.axhline(1, color='black', lw=3)

plt.legend("","");
# Define the minumum of periods to consider 

min_periods = 75 



# Calculate the volatility

vol = daily_pct_c.rolling(min_periods).std() * np.sqrt(min_periods) 



# Plot the volatility

vol.plot(figsize=(10, 4))



# Show the plot

plt.legend('', '');
AAPL.head()
# Initialize the short and long windows

short_window = 40

long_window = 100



# Initialize the `signals` DataFrame with the `signal` column

signals = pd.DataFrame(index=AAPL.index)

signals['signal'] = 0.0



# Create short simple moving average over the short window

signals['short_mavg'] = AAPL['close'].rolling(window=short_window, min_periods=1, center=False).mean()



# Create long simple moving average over the long window

signals['long_mavg'] = AAPL['close'].rolling(window=long_window, min_periods=1, center=False).mean()



# Create signals

signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 

                                            > signals['long_mavg'][short_window:], 1.0, 0.0)   



# Generate trading orders

signals['positions'] = signals['signal'].diff()
# Initialize the plot figure

fig = plt.figure(figsize=(10, 8))



# Add a subplot and label for y-axis

ax1 = fig.add_subplot(111,  ylabel='Price in $')



# Plot the closing price

AAPL['close'].plot(ax=ax1, color='grey', lw=2.)



# Plot the short and long moving averages

signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)



# Plot the buy signals

ax1.plot(signals.loc[signals.positions == 1.0].index, 

         signals.short_mavg[signals.positions == 1.0],

         '^', markersize=10, color='m')

         

# Plot the sell signals

ax1.plot(signals.loc[signals.positions == -1.0].index, 

         signals.short_mavg[signals.positions == -1.0],

         'v', markersize=10, color='k')

         

# Show the plot

plt.show()
# Set the initial capital

initial_capital= float(100000.0)



# Create a DataFrame `positions`

positions = pd.DataFrame(index=signals.index).fillna(0.0)



# Buy a 100 shares

positions['AAPL'] = 100*signals['signal']   

  

# Initialize the portfolio with value owned   

portfolio = positions.multiply(AAPL['close'], axis=0)



# Store the difference in shares owned 

pos_diff = positions.diff()



# Add `holdings` to portfolio

portfolio['holdings'] = (positions.multiply(AAPL['close'], axis=0)).sum(axis=1)



# Add `cash` to portfolio

portfolio['cash'] = initial_capital - (pos_diff.multiply(AAPL['close'], axis=0)).sum(axis=1).cumsum()   



# Add `total` to portfolio

portfolio['total'] = portfolio['cash'] + portfolio['holdings']



# Add `returns` to portfolio

portfolio['returns'] = portfolio['total'].pct_change()
fig = plt.figure(figsize=(10, 8))



ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')



# Plot the equity curve in dollars

portfolio['total'].plot(ax=ax1, lw=2.)



# Plot the "buy" trades against the equity curve

ax1.plot(portfolio.loc[signals.positions == 1.0].index, 

         portfolio.total[signals.positions == 1.0],

         '^', markersize=10, color='m')



# Plot the "sell" trades against the equity curve

ax1.plot(portfolio.loc[signals.positions == -1.0].index, 

         portfolio.total[signals.positions == -1.0],

         'v', markersize=10, color='k')



# Show the plot

plt.show()
# Isolate the returns of your strategy

returns = portfolio['returns']



# annualized Sharpe ratio

sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())



# Print the Sharpe ratio

print(np.round(sharpe_ratio,2))
# Define a trailing 252 trading day window

window = 252



# Calculate the max drawdown in the past window days for each day

rolling_max = AAPL['close'].rolling(window, min_periods=1).max()

daily_drawdown = AAPL['close']/rolling_max - 1.0



# Calculate the minimum (negative) daily drawdown

max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()



# Plot the results

daily_drawdown.plot()

max_daily_drawdown.plot()



# Show the plot

plt.show()
# Get the number of days in `aapl`

days = (AAPL.index[-1] - AAPL.index[0]).days



# Calculate the CAGR 

cagr = ((((AAPL['close'][-1]) / AAPL['close'][1])) ** (365.0/days)) - 1



# Print CAGR

print(np.round(cagr,3))
# Drop the columns

ph_df = AAPL.drop(['open', 'high', 'low','volume'], axis=1)

ph_df.reset_index(inplace=True)

ph_df.rename(columns={'close': 'y', 'date': 'ds'}, inplace=True)

ph_df['ds'] = pd.to_datetime(ph_df['ds'])

ph_df.head()

!pip3 uninstall --yes fbprophet

!pip3 install fbprophet --no-cache-dir --no-binary :all:
from fbprophet import Prophet

m = Prophet()

m.fit(ph_df)
# Create Future dates

future_prices = m.make_future_dataframe(periods=365)



# Predict Prices

forecast = m.predict(future_prices)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = m.plot(forecast)

ax1 = fig.add_subplot(111)

ax1.set_title("AAPL Stock Price Forecast", fontsize=16)

ax1.set_xlabel("Date", fontsize=12)

ax1.set_ylabel("Close Price", fontsize=12)

plt.show()
fig2 = m.plot_components(forecast)

plt.show()
# Monthly Data Predictions

m = Prophet(changepoint_prior_scale=0.01).fit(ph_df)

future = m.make_future_dataframe(periods=12, freq='M')

fcst = m.predict(future)

fig = m.plot(fcst)

plt.title("Monthly Prediction \n 1 year time frame", fontsize=16)

plt.xlabel("Date", fontsize=12)

plt.ylabel("Close Price", fontsize=12)



plt.show()
fig = m.plot_components(fcst)

plt.show()
from plotly import tools

import plotly.plotly as py

import plotly.figure_factory as ff

import plotly.tools as tls

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
trace = go.Ohlc(x=AAPL.index,

                open=AAPL['open'],

                high=AAPL['high'],

                low=AAPL['low'],

                close=AAPL['close'],

               increasing=dict(line=dict(color= '#58FA58')),

                decreasing=dict(line=dict(color= '#FA5858')))



layout = {

    'title': 'AAPL Historical Price',

    'xaxis': {'title': 'Date',

             'rangeslider': {'visible': False}},

    'yaxis': {'title': 'Stock Price (USD$)'},

    'shapes': [{

        'x0': '2018-12-31', 'x1': '2018-12-31',

        'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',

        'line': {'color': 'rgb(30,30,30)', 'width': 1}

    }],

    'annotations': [{

        'x': '2019-01-01', 'y': 0.05, 'xref': 'x', 'yref': 'paper',

        'showarrow': False, 'xanchor': 'left',

        'text': '2019 <br> starts'

    }]

}



data = [trace]



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='simple_ohlc')
# Moving Averages (10, 50 and 200)

AAPL['10_d_avg'] = AAPL.close.rolling(window=10).mean()

AAPL['50_d_avg'] = AAPL.close.rolling(window=50).mean()

AAPL['200_d_avg'] = AAPL.close.rolling(window=200).mean()

close_p = AAPL['close'].values.tolist()





# Variables to insert into plotly

ten_d = AAPL['10_d_avg'].values.tolist()

fifty_d = AAPL['50_d_avg'].values.tolist()

twoh_d = AAPL['200_d_avg'].values.tolist()

date = AAPL.index.values.tolist()



# Set date as index

#amzn_df = amzn_df.set_index('date')
fig = tls.make_subplots(rows=2, cols=1, shared_xaxes=True)



colors = ['#ff4500', '#92a1cf', '#6E6E6E']

avgs = ['10_d_avg', '50_d_avg', '200_d_avg']

# for i,c in zip(range(n),color):

#    ax1.plot(x, y,c=c)



for col, c in zip(avgs, colors):

    fig.append_trace({'x': AAPL.index, 'y': AAPL[col], 'type': 'scatter', 'name': col, 'line': {'color': c}}, 1, 1)

for col in ['close']:

    fig.append_trace({'x': AAPL.index, 'y': AAPL[col], 'type': 'scatter', 'name': 'Closing Price', 'line':{'color': '#01DF3A'}}, 2, 1)

    

fig['layout'].update(height=800,title='Relationship between MAs <br> and Closing Price',

                    paper_bgcolor='#F2DFCE', plot_bgcolor='#F2DFCE')

    

iplot(fig, filename='pandas/mixed-type subplots')
# Take off the date index

AAPL = AAPL.reset_index()



# Plotly

trace0 = go.Scatter(

    x = AAPL['date'],

    y = ten_d,

    name = '10-day MA',

    line = dict(

        color = ('#ff6347'),

        width = 4)

)

trace1 = go.Scatter(

    x = AAPL['date'],

    y = fifty_d,

    name = '50-day MA',

    line = dict(

        color = ('#92a1cf'),

        width = 4,

    dash="dot")

)

trace2 = go.Scatter(

    x = AAPL['date'],

    y = twoh_d,

    name = '200-day MA',

    line = dict(

        color = ('#2EF688'),

        width = 4,

        dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'

)



data = [trace0, trace1, trace2]





# Edit the layout

layout = dict(title = 'Moving Averages for AAPL',

              xaxis = dict(title = 'Date'),

              yaxis = dict(title = 'Price'),

#               annotations=[

#         dict(

#             x='2016-01-13',

#             y=657,

#             xref='x',

#             yref='y',

#             text='<i> First major decline <br> after 10-d crosses <br> 50-d moving average </i>',

#             showarrow=True,

#             arrowhead=5,

#             ax=5,

#             ay=-50

#         ), dict(

#         x = "2016-02-24",

#         y = 535,

#         text = "<i>Upward trend after <br> 10-day crosses <br>200-day moving average </i>",

#         textangle = 0,

#         ax = 50,

#         ay = 50,

#         font = dict(

#           color = "black",

#           size = 12

#         )

#         )],

              paper_bgcolor='#FFF9F5',

              plot_bgcolor='#FFF9F5'

              )



fig = dict(data=data, layout=layout)

iplot(fig, filename='styled-line')
df_MSFT = df_stocks[df_stocks['symbol'] == "MSFT"].copy()

df_MSFT.drop('symbol', axis=1, inplace=True)

df_MSFT.set_index('date', inplace=True)

df_MSFT.head()
def feature_creator(df):

    

    ## Current price

    lambd = df['close']

    

    ## Rolling mean

    rm_lambd = df['close'].rolling(window=60).mean()

    

    ## Rolling volatility

    rstd_lambd = (df['close'].rolling(window=60).std()) / lambd

    

    ## Trading range

    df_range = (df['low'] - df['high']) / df['high']

    rm_df_range = df_range.rolling(window=60).mean()

    

    ## Momentum

    momentum = (df['close'] - df['close'].shift(60)) / df['close']

    

    ## Cumulative returns

    cumula_ret_df = df['close']

    cumula_ret_df = (cumula_ret_df / cumula_ret_df.ix[0]) - 1

    

    ##  Distance from moving average

    dfrm = (df['close'] - rm_lambd) / df['close']

    

    ## Standard deviations from moving average

    stfrm = (df['close'] - rm_lambd) / df['close'].rolling(window=60).std()

    

    ## Daily returns

    daily_ret = df['close']

    daily_ret = (daily_ret / daily_ret.shift(1)) - 1

    daily_ret = daily_ret.rolling(window=60).mean()

    daily_ret = daily_ret.drop(daily_ret.index[0:60])

    

    ## Volatility momentum

    std_rat_1 = df['close'].rolling(window=60).std()

    std_rat_2 = df['close'].shift(1).rolling(window=60).std()

    std_rat = (std_rat_1 / std_rat_2) - 1

    

    ## ADX, DI_minus, DI_plus

    p_high = df['high'].shift(1).drop(df['high'].index[0])

    p_low = df['low'].shift(1).drop(df['low'].index[0])

    p_close = df['close'].shift(1).drop(df['close'].index[0])

    high = df['high'].drop(df['high'].index[0])

    low = df['low'].drop(df['low'].index[0])

    close = df['close'].drop(df['close'].index[0])

    UpMove = high - p_high

    DownMove = low - p_low

    up_more_down = UpMove > DownMove

    up_more_zero = UpMove > 0

    down_more_up = DownMove > UpMove

    down_more_zero = DownMove > 0

    data = pd.concat([UpMove, DownMove], axis=1)

    data = data.rename(columns={'High':'UpMove', 'Low':'DownMove'})

    data['DI_plus_one'] = up_more_down & up_more_zero

    data['DI_minus_one'] = down_more_up & down_more_zero

    data['DI_plus'] = (1 * data['DI_plus_one']) * UpMove

    data['DI_minus'] = (1 * data['DI_minus_one']) * DownMove

    data_x = data.drop(['DI_plus_one', 'DI_minus_one'], axis=1)

    DI_plus = data_x['DI_plus']

    DI_minus = data_x['DI_minus']

    TR1 = abs(high - low)

    TR2 = abs(high - p_close)

    TR3 = abs(low - p_close)

    TR_max = np.maximum(TR1, TR2)

    TR = np.maximum(TR_max, TR3)

    TR_ewma_x = TR.ewm(com=13, min_periods=28).mean()

    TR_ewma = TR_ewma_x.drop(TR_ewma_x.index[0:13])

    plus_div = DI_plus / TR_ewma

    minus_div = DI_minus / TR_ewma

    plus_DI_x = 100 * plus_div.ewm(com=13, min_periods=28).mean()

    plus_DI = plus_DI_x.drop(plus_DI_x.index[0:26])

    minus_DI_x = 100 * minus_div.ewm(com=13, min_periods=28).mean()

    minus_DI = minus_DI_x.drop(minus_DI_x.index[0:26])

    ADX_x = abs((plus_DI - minus_DI) / (plus_DI + minus_DI))

    ADX_xx = 100 * ADX_x.ewm(com=13, min_periods=28).mean()

    ADX = ADX_xx.drop(ADX_xx.index[0:26])

    m_plus_DI_xx = (plus_DI - plus_DI.shift(2)) / plus_DI

    m_minus_DI_xx = (minus_DI - minus_DI.shift(2)) / minus_DI

    m_plus_DI_x = m_plus_DI_xx.drop(m_plus_DI_xx.index[0:3])

    m_minus_DI_x = m_minus_DI_xx.drop(m_minus_DI_xx.index[0:3])

    

    ## Volume / 10 000

    volume = df['volume'] / 10000

    

    ## Change in volume

    m_volume = (df['volume'] - df['volume'].shift(60)) / df['volume']

        

    ## Trig angle

    period = 60

    x1 = df['close'].shift(period)

    x1 = x1.drop(x1.index[0:period])

    x3 = df['close'].drop(df['close'].index[0:6])

    x2_max = df['close'].rolling(window=period).max()

    x2_min = df['close'].rolling(window=period).min()

    f1 = (df['close'] - x2_max)

    f2 = (df['close'] - x2_min)

    f_max = (f1 > f2).astype(int)

    f_min = (f2 > f1).astype(int)

    zz1 = f_max * x2_max

    zz2 = f_min * x2_min

    x2 = zz1 + zz2

    x2 = x2.drop(x2.index[0:6])

    x_len = np.sqrt((np.square(x2 - x1)) + np.square(period))

    y_len = np.sqrt((np.square(x2 - x3)) + np.square(period))

    y_len = y_len.drop(y_len.index[0:period])

    x_theta = np.arcsin(period / x_len)

    y_theta = np.arcsin(period / y_len)

    z1 = 90 - (np.arcsin(period / x_len))

    z2 = 90 - (np.arcsin(period / y_len))

    trig_x = 180 - z1 - z2

    min_value = x3 > x2

    max_value = x2 > x3

    mini = min_value.drop(min_value.index[0:5]).astype(int)

    maxi = max_value.drop(max_value.index[0:5]).astype(int) * - 1

    mask = mini + maxi

    trig = trig_x * mask



    features = [ lambd, rm_lambd, rstd_lambd, rm_df_range, momentum, cumula_ret_df, dfrm, stfrm, daily_ret,

               std_rat, m_plus_DI_x, m_minus_DI_x, ADX, volume, m_volume, trig]

    features = pd.concat(features, axis=1)

    features.columns = ['current value', 'rolling mean', 'volatility', 'trading range', 'momentum', 'cumulative value',

                       'distance from moving average', 'standard deviations from moving average', 'daily change', 'change in volatility', 'Change in + directional movement', 'Change in - directional movement', 'Average Directional Index', 'volume', 'change in volume', 'angle']

    features = features.drop(features.index[0:82])

    features = features.drop(features.index[-60:])

    

    return features
def target_creator(df):

    target = df['close'].shift(-60)

    current_close = df['close']

    frames = [current_close, target]

    out = pd.concat(frames, axis=1)

    out.columns = ['current_close','target']

    out = out.drop(out.index[0:82])

    out = out.drop(out.index[-60:])

    return out
X, y = feature_creator(df_MSFT), target_creator(df_MSFT)['target']
plt.plot(X['current value'], label='current price')

plt.plot(y, label='target price in 60 trading days')

plt.legend()

sns.despine();