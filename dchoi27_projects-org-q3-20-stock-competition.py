!pip install 'yfinance'

import yfinance as yf

from datetime import date

import pandas as pd

import matplotlib.pyplot as plt
def validate_allocation(name, allocation):

    short = allocation['short']

    if any(x>0 for x in short.values()) == True:

        raise ValueError('Not all positions in ' + name + '\'s short allocation are actually short')

    if sum(short.values()) < -2000:

        raise ValueError(name + '\'s short allocation less than -2000')



    long = allocation['long']

    if any(x<0 for x in long.values()) == True:

        raise ValueError('Not all positions in ' + name + '\'s long allocation are actually long')

    if sum(long.values()) != (10000 + -1 * sum(short.values())):

        raise ValueError(name + '\'s long allocation does not sum to 10000 + short allocation')

    

    print('all of ' + name + '\'s allocations validated')
# input portfolio allocations



david_allocation = {

    'long': {

        'DPZ': 4000,

        'LUV': 2500,

        'AMC': 1500,

        'PANW': 2000,

        'SPLK': 2000

    }, 

    'short': {

        'IGT': -2000

    }

}

ignacio_allocation = {

    'long': {

        'AAPL': 2500,

        'FB': 500,

        'NVDA': 2250,

        'CRM': 1500,

        'PYPL': 1500,

        'SEDG': 2000,

        'FLR': 1750

    }, 

    'short': {

        'BAYRY': -2000

    }

}

kwray_allocation = {

    'long': {

        'AGQ': 3000,

        'PTN': 5000,

        'PLT': 2000,

        'SSSS': 1000,

        'HMY': 1000

    }, 

    'short': {

        'GME': -2000

    }

}

mike_allocation = {

    'long': {

        'PSQ': 12000,

    }, 

    'short': {

        'AAPL': -2000

    }

}

steve_allocation = {

    'long': {

        'UNH': 2400,

        'MSFT': 2400,

        'GO': 2400,

        'MRNA': 2400,

        'THO': 2400

    }, 

    'short': {

        'TSLA': -2000

    }

}



s_and_p = {

    'long': {

        'SPY': 10000

    },

    'short': {

    }

}



#build portfolio for all participants

all_allocations = {

    'david': david_allocation,

    'ignacio': ignacio_allocation,

    'kwray': kwray_allocation,

    'mike': mike_allocation,

    'steve': steve_allocation,

    's&p500': s_and_p

}



for name, allocation in all_allocations.items():

    validate_allocation(name, allocation)
# input start date of competition, and how far back graphs should go back here

competition_start_date = '2020-08-05'

competition_end_date = '2020-10-01'

graph_start_date = '2020-07-01'



portfolio_performance = pd.DataFrame()

portfolio_breakdown = dict.fromkeys(all_allocations.keys())



for name, allocation in all_allocations.items():

    # initialize blank data frames

    historical_portfolio_value = yf.Ticker('AAPL').history(

            period = '1d', start = graph_start_date, end = date.today())['Close'] * 0

    individual_stock_performance = pd.DataFrame(index = historical_portfolio_value.index)

    

    for stock_name, position in allocation['long'].items():

        stock_price = yf.Ticker(stock_name).history(

            period = '1d', start = graph_start_date, end = date.today())['Close']

        stock_historical_position = position * (stock_price / stock_price[competition_start_date])

        historical_portfolio_value += stock_historical_position

        individual_stock_performance[stock_name] = stock_historical_position

    

    for stock_name, position in allocation['short'].items():

        stock_price = yf.Ticker(stock_name).history(

            period = '1d', start = graph_start_date, end = date.today())['Close']

        stock_historical_position = position * (stock_price / stock_price[competition_start_date])

        historical_portfolio_value += stock_historical_position

        individual_stock_performance[stock_name] = stock_historical_position

        

    portfolio_performance[name] = historical_portfolio_value

    portfolio_breakdown[name] = individual_stock_performance



portfolio_performance[competition_start_date:competition_end_date]
# plot portfolio vs. portfolio performance



ax = portfolio_performance.plot.line()

ax.axvline(competition_start_date, color = 'black', linestyle = '--')
# plot individual portfolio performance



plots_vertical = int(len(portfolio_breakdown) / 2) + 1

fig, axes = plt.subplots(plots_vertical, 2)

fig.set_figheight(10 * plots_vertical)

fig.set_figwidth(20)

i = 0

for name, portfolio in portfolio_breakdown.items():

    ax = portfolio.plot.line(ax=axes[int(i/2),i % 2])

    axes[int(i/2),i % 2].set_title(name)

    i+=1

    ax.axvline(competition_start_date, color = 'black', linestyle = '--')