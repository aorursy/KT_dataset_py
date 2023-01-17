import numpy as np 

import pandas as pd 

from os import listdir



import datetime

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
# Make the default figures a bit bigger

plt.rcParams['figure.figsize'] = (5,3) 

plt.rcParams["figure.dpi"] = 120 



sns.set(style="ticks")

sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 5})

greek_salad = ['#D0D3C5', '#56B1BF', '#08708A', '#D73A31', '#032B2F']

sns.set_palette(greek_salad)
# https://www.nasdaq.com/screening/company-list.aspx

nasdaq = pd.read_csv('../input/nasdaq-company-list/companylist.csv')

cols = ['Symbol', 'Name', 'MarketCap', 'Sector']

nasdaq = nasdaq[cols]

nasdaq = nasdaq.drop_duplicates(subset=['Name'], keep='first')

nasdaq = nasdaq[nasdaq['MarketCap'] >= 1e9]

print(nasdaq.shape)

nasdaq.sort_values(by='MarketCap', ascending=False).head(10)
def find_csv_filenames( path_to_dir, suffix=".csv" ):

    filenames = listdir(path_to_dir)

    return [ filename for filename in filenames if filename.endswith( suffix ) ]



path_to_dir = '../input/amex-nyse-nasdaq-stock-histories/fh_20190217/full_history/'

filenames = find_csv_filenames(path_to_dir)
%%time 

# Create Empty Dataframe

stock_final = pd.DataFrame()



for i in range(len(list(nasdaq['Symbol']))): #filenames

    #print(i)    

    try:

        stock=[]

        stock = pd.read_csv(path_to_dir+list(nasdaq['Symbol'])[i]+'.csv')

        stock['name'] = list(nasdaq['Symbol'])[i] #filenames[i].replace(".csv", "")

        # Data starting from 2015

        stock['date'] = pd.to_datetime(stock['date'])

        stock = stock[stock.date >= '2016-01-01']

        stock_final = pd.DataFrame.append(stock_final, stock, sort=False)

    

    except Exception:

        i = i+1     
print("Available tickers", stock_final.name.nunique())

display(stock_final.sample(3))
cols = ['date', 'adjclose', 'name']

df_close = stock_final[cols].pivot(index='date', columns='name', values='adjclose')



cols = ['date', 'volume', 'name']

df_volume = stock_final[cols].pivot(index='date', columns='name', values='volume')



print('Dataset shape:',df_close.shape)

display(df_close.tail(3))



print('Dataset shape:',df_volume.shape)

display(df_volume.tail(3))
percent_missing = pd.DataFrame(df_close.isnull().sum() * 100 / len(df_close))

percent_missing.columns = ['percent_missing']

percent_missing.sort_values('percent_missing', inplace=True, ascending=False)



percent_missing_plot = pd.DataFrame(percent_missing.reset_index().groupby('percent_missing').size())

percent_missing_plot.reset_index(inplace=True)

percent_missing_plot.columns = ['percent_missing', 'count']



ax = sns.scatterplot(x='percent_missing', y='count', data=percent_missing_plot, color=greek_salad[2])

ax.set_yscale('log')

ax.set_ylabel('Number of tickers')

ax.set_xlabel('Missing Data (%)')

sns.despine()
complete_data_tickers = percent_missing[percent_missing['percent_missing'] == 0].index

df = df_close[complete_data_tickers].head()



print("Available tickers", df.shape[1])

display(df.sample(3))
df_pct_change = df.pct_change()

df_pct_change.head(3)
complete_data_tickers = percent_missing[percent_missing['percent_missing'] == 0].index

df_volume = df_volume[complete_data_tickers].head()



df_vol_change = df_volume.pct_change()

df_vol_change.head(3)
plt.figure(figsize=(6,6))

corr = df_pct_change.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, cmap=sns.diverging_palette(20,220, n=11), center=0)

    ax.set_yticklabels('')

    ax.set_xticklabels('')

    ax.set_ylabel('')

    ax.set_xlabel('')

    sns.despine()

    plt.tight_layout()
plt.figure(figsize=(6,6))

corr = df_vol_change.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, cmap=sns.diverging_palette(20,220, n=11), center=0)

    ax.set_yticklabels('')

    ax.set_xticklabels('')

    ax.set_ylabel('')

    ax.set_xlabel('')

    sns.despine()

    plt.tight_layout()
MSFT = stock_final[stock_final.name == 'MSFT'].copy()

MSFT.set_index('date', inplace=True)

MSFT.head()
from plotly import tools

import plotly.plotly as py

import plotly.figure_factory as ff

import plotly.tools as tls

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



trace = go.Ohlc(x=MSFT.index,

                open=MSFT['open'],

                high=MSFT['high'],

                low=MSFT['low'],

                close=MSFT['close'],

               increasing=dict(line=dict(color= '#58FA58')),

                decreasing=dict(line=dict(color= '#FA5858')))



layout = {

    'title': 'MSFT Historical Price',

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
# Drop the columns

ph_df = MSFT.drop(['open', 'high', 'low','volume', 'adjclose', 'name'], axis=1)

ph_df.reset_index(inplace=True)

ph_df.rename(columns={'close': 'y', 'date': 'ds'}, inplace=True)

ph_df['ds'] = pd.to_datetime(ph_df['ds'])

ph_df['y'] = np.log1p(ph_df['y'])

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

ax1.set_title("MSFT Stock Price Forecast", fontsize=16)

ax1.set_xlabel("Date", fontsize=12)

ax1.set_ylabel("$log(1 + Close Price)$", fontsize=12)

sns.despine()

plt.tight_layout()
# fig2 = m.plot_components(forecast)

# plt.show()
# Monthly Data Predictions

m = Prophet(changepoint_prior_scale=0.01).fit(ph_df)

future = m.make_future_dataframe(periods=12, freq='M')

fcst = m.predict(future)

fig = m.plot(fcst)

plt.title("Monthly Prediction \n 1 year time frame", fontsize=16)

plt.xlabel("Date", fontsize=12)

plt.ylabel("$log(1+Close Price)$", fontsize=12)

sns.despine()

plt.tight_layout()
stock_max = np.round(np.expm1(fcst.tail(12)['yhat'].max()), 2)

stock_min = np.round(np.expm1(fcst.tail(12)['yhat'].min()), 2)

stock_current = np.expm1(ph_df.sort_values(by='ds').tail(1)['y'].values)



gain = (stock_max - stock_current) / stock_current

loss = (stock_current - stock_min) / stock_current



print('Current price:', np.round(stock_current,2), '$')

print('Expected High:', np.round(stock_max,2), '$')

print('Expected Low:', np.round(stock_min,2), '$')

print('Expected profit:', np.round(gain*100,2), '%')

print('Expected loss:', np.round(loss*100,2), '%')
!pip3 install fix_yahoo_finance --upgrade --no-cache-dir
from pandas_datareader import data as pdr

import fix_yahoo_finance as yf

from datetime import date



yf.pdr_override() 



end = date.today()

DJI = pdr.get_data_yahoo("^DJI", start="2016-01-01", end=end)
DJI.tail()
from plotly import tools

import plotly.plotly as py

import plotly.figure_factory as ff

import plotly.tools as tls

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



trace = go.Ohlc(x=DJI.index,

                open=DJI['Open'],

                high=DJI['High'],

                low=DJI['Low'],

                close=DJI['Close'],

               increasing=dict(line=dict(color= '#58FA58')),

                decreasing=dict(line=dict(color= '#FA5858')))



layout = {

    'title': 'DJI Historical Price',

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
# Drop the columns

ph_df = DJI.drop(['Open', 'High', 'Low','Volume', 'Adj Close'], axis=1)

ph_df.reset_index(inplace=True)

ph_df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

ph_df['ds'] = pd.to_datetime(ph_df['ds'])

ph_df['y'] = np.log1p(ph_df['y'])

ph_df.head()



# Monthly Data Predictions

m = Prophet(changepoint_prior_scale=0.01).fit(ph_df)

future = m.make_future_dataframe(periods=12, freq='M')

fcst = m.predict(future)

fig = m.plot(fcst)

plt.title("Monthly Prediction for DJI Index \n 1 year time frame", fontsize=16)

plt.xlabel("Date", fontsize=12)

plt.ylabel("$log(1+Close)$", fontsize=12)

sns.despine()

plt.tight_layout()
stock_max = np.round(np.expm1(fcst.tail(12)['yhat'].max()), 2)

stock_min = np.round(np.expm1(fcst.tail(12)['yhat'].min()), 2)

stock_current = np.expm1(ph_df.sort_values(by='ds').tail(1)['y'].values)



DJI_gain = (stock_max - stock_current) / stock_current

DJI_loss = (stock_current - stock_min) / stock_current



print('Current :', np.round(stock_current,2))

print('Expected High:', np.round(stock_max,2))

print('Expected Low:', np.round(stock_min,2))

print('Expected rise:', np.round(DJI_gain*100,2), '%')

print('Expected fall:', np.round(DJI_loss*100,2), '%')
%%time

df_gains = pd.DataFrame()

i = 0

for ticker in df_close.columns:

    tmp = pd.DataFrame()

    ticker = df_close.columns[i]

    ph_df = pd.DataFrame(df_close[ticker].copy())

    ph_df.reset_index(inplace=True)

    ph_df.rename(columns={ticker: 'y', 'date': 'ds'}, inplace=True)

    ph_df['ds'] = pd.to_datetime(ph_df['ds'])

    ph_df['y'] = np.log1p(ph_df['y'])



    m = Prophet(changepoint_prior_scale=0.01).fit(ph_df)

    future = m.make_future_dataframe(periods=12, freq='M')

    fcst = m.predict(future)

    

    stock_max = np.round(np.expm1(fcst.tail(12)['yhat'].max()), 2)

    stock_min = np.round(np.expm1(fcst.tail(12)['yhat'].min()), 2)

    stock_current = np.expm1(ph_df.sort_values(by='ds').tail(1)['y'].values)



    gain = (stock_max - stock_current) / stock_current

    loss = (stock_current - stock_min) / stock_current

    tmp = pd.DataFrame([ticker, gain, loss]).T

    t = [('ticker', ticker),

         ('gain', gain),

         ('loss', loss)]

    tmp = pd.DataFrame.from_items(t)

    df_gains = df_gains.append(tmp)

    i = i+1

    
df_gains = df_gains.loc[(df_gains['gain'] >= DJI_gain[0])]

df_gains = df_gains.loc[(df_gains['loss'] <= DJI_loss[0])]

df_gains.sample(5)
fig = figure(num=None, figsize=(12, 4), dpi=120, facecolor='w', edgecolor='k')



plt.subplot(1, 1, 1)

ax1 = sns.distplot(df_gains['gain'].dropna()*100, bins=50, color=greek_salad[2]);

#ax1.set_xlim(0, 400)

ax1.set_xlabel('Gain (%)', weight='bold')

ax1.set_ylabel('Density', weight = 'bold')

ax1.set_title('Distribution of expected 1 year gain')

sns.despine()

plt.tight_layout();
# ## Distribution of expected loss

# fig = figure(num=None, figsize=(12, 4), dpi=120, facecolor='w', edgecolor='k')



# plt.subplot(1, 1, 1)

# ax1 = sns.distplot(df_gains['loss'].dropna()*100, bins=50, color=greek_salad[3]);

# #ax1.set_xlim(0, 400)

# ax1.set_xlabel('Loss (%)', weight='bold')

# ax1.set_ylabel('Density', weight = 'bold')

# ax1.set_title('Distribution of expected 1 year loss')

# sns.despine()

# plt.tight_layout();
df_selected_stocks = pd.merge(df_gains, nasdaq, how='inner', left_on='ticker', right_on='Symbol')

cols = ['ticker', 'gain', 'Name', 'MarketCap', 'Sector']



df_selected_stocks = df_selected_stocks[cols]

df_selected_stocks.to_csv('selected_stocks.csv', sep=',', encoding='utf-8')

df_selected_stocks.sample(5)
f = {'gain':['median'], 'MarketCap':['sum'], 'Name':['count']}



ratios = df_selected_stocks.groupby('Sector').agg(f)

ratios.columns = ratios.columns.get_level_values(0)

ratios = ratios.reset_index()

ratios = ratios.sort_values('gain', ascending=False)



fig = figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')



plt.subplot(1, 3, 1)

ax1 = sns.barplot(x="Name", y="Sector", data=ratios, palette=("Greys_d"))

ax1.set_xlabel('Number of companies', weight='bold')

ax1.set_ylabel('Sector', weight = 'bold')

ax1.set_title('Sector breakdown\n')



plt.subplot(1, 3, 2)

ax2 = sns.barplot(x="MarketCap", y="Sector", data=ratios, palette=("Greens_d"))

ax2.set_xlabel('Total Market Cap', weight='bold')

ax2.set_ylabel('')

ax2.set_yticks([])



plt.subplot(1, 3, 3)

ax2 = sns.barplot(x="gain", y="Sector", data=ratios, palette=("Greens_d"))

ax2.set_xlabel('Median Gain', weight='bold')

ax2.set_ylabel('')

ax2.set_yticks([])



sns.despine()

plt.tight_layout();