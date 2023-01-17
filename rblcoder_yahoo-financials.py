!pip install yahoofinancials
from yahoofinancials import YahooFinancials



ticker = 'AAPL'

yahoo_financials = YahooFinancials(ticker)
historical_stock_prices = yahoo_financials.get_historical_price_data('2008-09-15', '2019-12-31', 'weekly')

historical_stock_prices
import pandas as pd

pd.json_normalize(historical_stock_prices['AAPL']['prices'])
!pip install backtrader
# https://github.com/mementum/backtrader



from datetime import datetime

import backtrader as bt



class SmaCross(bt.SignalStrategy):

    def __init__(self):

        sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)

        crossover = bt.ind.CrossOver(sma1, sma2)

        self.signal_add(bt.SIGNAL_LONG, crossover)



cerebro = bt.Cerebro()

cerebro.addstrategy(SmaCross)



data0 = bt.feeds.YahooFinanceData(dataname='AAPL', fromdate=datetime(2011, 1, 1),

                                  todate=datetime(2019, 12, 31))

cerebro.adddata(data0)



cerebro.run()

cerebro.plot()