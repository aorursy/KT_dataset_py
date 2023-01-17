import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

from datetime import datetime

from datetime import date



# To read our stock information from Yahoo:

import pandas_datareader as pdr



sns.set_style('whitegrid')
# Identify our stocks - four tech, four retail

ticker_list = ['AAPL','GOOG','MSFT','AMZN','COST','WBA','WMT','HD']



# Set timeframe

end = date(2020,8,19)

start = datetime(end.year-1,end.month,end.day)



# Pull stock info and assign ticker as name of dataframe

for stock in ticker_list:

    globals()[stock] = pdr.get_data_yahoo(stock,start,end)
AAPL.describe()
AAPL.info()
fig, axes = plt.subplots(4,2,figsize=(16,16))

frames = [AAPL,GOOG,MSFT,AMZN,COST,WBA,WMT,HD]



for i, ax in enumerate(axes.flatten()):

    frames[i]['Adj Close'].plot(legend=True,ax=ax)

    ax.set_title(label=ticker_list[i], fontsize=16, fontweight='bold', pad=10)

    ax.set_xlabel(None)

    

plt.tight_layout(h_pad=2)
fig, axes = plt.subplots(4,2,figsize=(16,16))

frames = [AAPL,GOOG,MSFT,AMZN,COST,WBA,WMT,HD]



for i, ax in enumerate(axes.flatten()):

    frames[i]['Volume'].plot(legend=True,ax=ax)

    ax.set_title(label=ticker_list[i], fontsize=16, fontweight='bold', pad=10)

    ax.set_xlabel(None)

    

plt.tight_layout(h_pad=2)
fig, axes = plt.subplots(4,2,figsize=(20,16))

ma_size = [10,20,50]



for i, ax in enumerate(axes.flatten()):

    for ma in ma_size:

        col = "MA for %s days" %(str(ma))

        frames[i][col] = frames[i]['Adj Close'].rolling(ma).mean()

    frames[i][['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(legend=True,ax=ax)

    ax.set_title(label=ticker_list[i], fontsize=16, fontweight='bold', pad=10)

    ax.set(xlabel=None)

        

plt.tight_layout(h_pad=2)
fig, axes = plt.subplots(4,2,figsize=(20,16))



for i, ax in enumerate(axes.flatten()):

    frames[i]['Daily Return'] = frames[i]['Adj Close'].pct_change()

    frames[i]['Daily Return'].plot(legend=True,linestyle='--',marker='o', ax=ax)

    ax.set_title(label=ticker_list[i], fontsize=16, fontweight='bold', pad=10)

    ax.set(xlabel=None)

    

plt.tight_layout(h_pad=2)
fig = plt.figure(figsize=(16,8))

g = fig.add_gridspec(2,4)

ax1 = fig.add_subplot(g[0, 0:1])

ax2 = fig.add_subplot(g[0, 1:2])

ax3 = fig.add_subplot(g[0, 2:3])

ax4 = fig.add_subplot(g[0, 3:4])

ax5 = fig.add_subplot(g[1, 0:1])

ax6 = fig.add_subplot(g[1, 1:2])

ax7 = fig.add_subplot(g[1, 2:3])

ax8 = fig.add_subplot(g[1, 3:4])



axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]

colors = ['Red','Orange','y','Green','Blue','Indigo','Violet','Brown']



for i, ax in enumerate(axes):

    sns.distplot(frames[i]['Daily Return'].dropna(),bins=100, ax=ax,

             kde_kws={'color':'darkolivegreen','label':'Kde','gridsize':1000,'linewidth':2},

             hist_kws={'color':'goldenrod','label':"Histogram",'edgecolor':'darkslategray'})

    ax.set_title(label=ticker_list[i], fontsize=16, fontweight='bold', pad=10)

    ax.set(xlabel=None)



plt.tight_layout(h_pad=2)
Adj_Close_df = pdr.get_data_yahoo(ticker_list,start,end)['Adj Close']

returns_df = Adj_Close_df.pct_change()

returns_df.head()
# Define corrfunc to plot pearson coefficient in top left corner of each plot

from scipy.stats import pearsonr

def corrfunc(x,y,ax=None, **kws):

    r, _ = pearsonr(x, y)

    ax = plt.gca()

    # Unicode for lowercase rho (ρ)

    rho = '\u03C1'

    ax.annotate(f'{rho} = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)



# Plot each pair's returns to see their correlation

g = sns.PairGrid(returns_df.dropna())

g.map_lower(corrfunc)

g.map_lower(plt.scatter)

g.map_upper(sns.kdeplot,cmap='Blues')

g.map_diag(plt.hist)

#plt.show()
plt.figure(figsize=(10,8))

sns.heatmap(returns_df.corr(),annot=True,square=True,cmap='Blues')
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,8),sharex=True)



def to_percent(y,position):

    return str(str(int(round(y*100,0)))+"%")



# Plot Annual Growth (%)

normalized_df = Adj_Close_df.copy()

for ticker in ticker_list:

    baseline = normalized_df[ticker].iloc[0]

    normalized_df[ticker] = normalized_df[ticker] / baseline - 1

normalized_df.plot(legend=True, ax=ax1)

ax1.set_title(label='Annual Growth (%)', fontsize=16, fontweight='bold', pad=10)

ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))



# Plot Growth During Corona Pandemic (%)

normalized_corona_df = Adj_Close_df.copy().loc['2020-02-20':]

for ticker in ticker_list:

    baseline = normalized_corona_df[ticker].iloc[0]

    normalized_corona_df[ticker] = normalized_corona_df[ticker] / baseline - 1

normalized_corona_df.plot(legend=True, ax=ax2)

ax2.set_title(label='Growth During Corona Pandemic (%)', fontsize=16, fontweight='bold', pad=10)

ax2.set(xlabel=None)

ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))
returns = returns_df.dropna()

area = np.pi*20



plt.figure(figsize=(10,8))

plt.scatter(returns.mean(), returns.std(), s=50)



plt.xlim(-.002,.005)

plt.ylim(.015,.03)

plt.xlabel('Expected Daily Return')

plt.ylabel('Standard Deviation')



for label, x, y in zip(returns.columns, returns.mean(), returns.std()):

    plt.annotate(

        label,

        xy = (x,y), xytext = (25,25),

        textcoords = 'offset points', ha = 'right', va = 'center',

        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-.5', ec='purple'))

plt.title(label='Daily Returns: Mean vs Standard Deviation', fontsize=16, fontweight='bold', pad=10)
fig = plt.figure(figsize=(16,8))

g = fig.add_gridspec(2,4)

ax1 = fig.add_subplot(g[0, 0:1])

ax2 = fig.add_subplot(g[0, 1:2])

ax3 = fig.add_subplot(g[0, 2:3])

ax4 = fig.add_subplot(g[0, 3:4])

ax5 = fig.add_subplot(g[1, 0:1])

ax6 = fig.add_subplot(g[1, 1:2])

ax7 = fig.add_subplot(g[1, 2:3])

ax8 = fig.add_subplot(g[1, 3:4])



axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]



for i, ax in enumerate(axes):

    data = returns[ticker_list[i]]

    std = data.std() 

    ax.text(min(data)*1.1, 45, s=f'Avg Daily Return = {(data.mean()):.2%}')

    ax.text(min(data)*1.1, 55, s=f'95%_ci = {(data.quantile(.05)):.2%}')

    ax.text(min(data)*1.1, 50, s=f'99%_ci = {(data.quantile(.01)):.2%}')

    sns.distplot(data, bins=100, ax=ax,

             kde_kws={'color':'darkolivegreen','gridsize':1000,'linewidth':2},

             hist_kws={'color':'goldenrod','edgecolor':'darkslategray'})

    ax.set(ylim=[0,60],title=ticker_list[i],xlabel=None)
def monte_carlo(start_price, days, mu, sigma):

    price = np.zeros(days)

    price[0] = start_price

    change = np.zeros(days)



    for i in range(1,days):

        change[i] = np.random.normal(loc=mu, scale=sigma)

        price[i] = price[i-1] * (1 + (change[i]))

        

    return price
fig, axes = plt.subplots(4,2,figsize=(16,24))



for i, ax in enumerate(axes.flatten()):

    days = 365

    mu = returns.mean()[ticker_list[i]]

    sigma = returns.std()[ticker_list[i]]

    start_price = Adj_Close_df[ticker_list[i]][0]

    

    for runs in range(100):

        ax.plot(monte_carlo(start_price,days,mu,sigma))



    ax.set(title=ticker_list[i])

    ax.set(xlabel='Days',ylabel='Price')
runs = 10000

simulations = np.zeros(runs)

for run in range(runs):

    simulations[run] = monte_carlo(start_price,days,mu,sigma)[days-1];



fig, axes = plt.subplots(4,2,figsize=(16,24))

colors = ['Red','Orange','y','Green','Blue','Indigo','Violet','Brown']



for i, ax in enumerate(axes.flatten()):

    runs = 10000

    simulations = np.zeros(runs)

    ticker = ticker_list[i]

    days = 365

    mu = returns.mean()[ticker]

    sigma = returns.std()[ticker]

    start_price = Adj_Close_df[ticker][0]

    for run in range(runs):

        simulations[run] = monte_carlo(start_price,days,mu,sigma)[days-1];



    q = np.percentile(simulations, 1)

    ax.hist(simulations, bins=200, color = colors[i])



    ax.text(0.7, 0.9, s="Start price: $%.2f" %start_price, transform=ax.transAxes)

    ax.text(0.7, 0.85, "Mean final price: $%.2f" % simulations.mean(), transform=ax.transAxes)

    ax.text(0.7, 0.8, "VaR(0.99): $%.2f" % (start_price - q), transform=ax.transAxes)

    ax.text(0.1, 0.9, "q(0.99): $%.2f" % q, transform=ax.transAxes)



    ax.axvline(x=q, linewidth=4, color='r')

    ax.set_title("Final price distribution for %s Stock after %s days" % (ticker,days), weight='bold')