!pip install backtrader
import backtrader as bt

import datetime

from matplotlib.dates import (HOURS_PER_DAY, MIN_PER_HOUR, SEC_PER_MIN,

                              MONTHS_PER_YEAR, DAYS_PER_WEEK,

                              SEC_PER_HOUR, SEC_PER_DAY,

                              num2date, rrulewrapper, YearLocator,

                              MicrosecondLocator)





class TestStrategy(bt.Strategy):



    def log(self, txt, dt=None):

        ''' Logging function fot this strategy'''

        dt = dt or self.datas[0].datetime.date(0)

        print('%s, %s' % (dt.isoformat(), txt))



    def __init__(self):

        # Keep a reference to the "close" line in the data[0] dataseries

        self.dataclose = self.datas[0].close

        self.order = None





    def notify_order(self, order):

        if order.status in [order.Submitted, order.Accepted]:

            return

        

        if order.status in [order.Completed]:

            if order.isbuy():

                self.log('BUY EXECUTED {}'.format(order.executed.price))

            elif order.issell():

                self.log('SELL EXECUTED {}'.format(order.executed.price))

        

            self.bar_executed = len(self)



        self.order = None





    def next(self):

        # Simply log the closing price of the series from the reference

        self.log('Close, %.2f' % self.dataclose[0])



        if self.order:

            return



        if not self.position:

            if self.dataclose[0] < self.dataclose[-1]:

                # current close less than previous close



                if self.dataclose[-1] < self.dataclose[-2]:

                    # previous close less than the previous close



                    # BUY, BUY, BUY!!! (with all possible default parameters)

                    self.log('BUY CREATE, %.2f' % self.dataclose[0])

                    self.order = self.buy()

        else:

            if len(self) >= (self.bar_executed + 5):

                self.log('SELL CREATED {}'.format(self.dataclose[0]))

                self.order = self.sell()

                



                

class Trader:



    cerebro = bt.Cerebro()



    cerebro.broker.set_cash(1000000)





    data = bt.feeds.YahooFinanceCSVData(dataname='../input/historical-amazon-stock-prices/AMZN.csv',

                                        fromdate=datetime.datetime(2020, 1, 1),

                                        todate=datetime.datetime(2020, 9, 23),

                                        reverse=False)

    cerebro.adddata(data)



    cerebro.addstrategy(TestStrategy)

    cerebro.addsizer(bt.sizers.FixedSize, stake=300)



    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())



    cerebro.run()



    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())



    #Save output

    

    figure = cerebro.plot(style ='candlebars')[0][0]

    figure.savefig('result.png')