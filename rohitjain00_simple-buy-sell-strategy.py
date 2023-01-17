# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

!pip install wallstreet

!pip install yfinance --upgrade --no-cache-dir

!pip install parallel-execute
from pexecute.process import ProcessLoom

import yfinance as yf

from datetime import datetime

import statistics

from wallstreet import Stock





class Volatility:

    def __init__(self, ticker, expiration):

        """

        :param ticker: ticker of the required stock's option

        :param expiration: expiration date object

        """

        self.ticker = ticker

        self.expiration = expiration

        self.stock = Stock(self.ticker)

        self.data = self._get_data()

        self.yfinance = yf.Ticker(self.ticker).option_chain(self.expiration.strftime("%Y-%m-%d"))

        calculated_std, calculated_mean = self.get_std_mean_of_volatility()

        self.calculated_volatility_std = calculated_std

        self.calculated_volatility_mean = calculated_mean



    def _get_data(self):

        return self.stock.historical(days_back=60, frequency='d')



    def is_option_overpriced(self, option):

        """

        calculates the avg volatility for the past year and calculate if the current option's implied volatility is more or less and in turn provides if the option is underpriced or overpriced

        :return:

        """

        index = 0

        if option['Option_type'] == 'Put':

            index = 1

        derived_volatility = self.calculated_volatility_mean

        yfinance_options = self.yfinance[index]

        yfinance_option_implied_volatility = yfinance_options.loc[yfinance_options['contractSymbol'] == option['code']][

            'impliedVolatility']

        provided_volatility = yfinance_option_implied_volatility

        return derived_volatility < provided_volatility



    def _day_difference(self, date1, date2):

        return (date2 - date1).days



    def get_std_mean_of_volatility(self):

        days_to_expire = self._day_difference(datetime.now().date(), self.expiration)

        volatility = []

        for dfIndex in range(0, len(self.data) - days_to_expire):

            temp_lis = []

            for day in range(0, days_to_expire):

                temp_lis.append(self.data.iloc[dfIndex + day]['Adj Close'])

            if len(temp_lis) < 1:

                continue

            volatility.append(statistics.pstdev(temp_lis) / statistics.mean(temp_lis))

        return statistics.pstdev(volatility), self._get_mean(volatility)



    def _get_mean(self, lis):

        """

        Can further be changed to exponential moving average

        :param lis:

        :return:

        """

        return statistics.mean(lis)
from wallstreet import Call, Put

from pexecute.process import ProcessLoom





class Options:

    def __init__(self, ticker, expiration):

        self.expiration = expiration

        self.ticker = ticker

        self.data = self._get_option_data()



    def _get_option_data(self):

        try:

            all_options = []

            loom = ProcessLoom(max_runner_cap=10)

            loom.add_function(self.call_lis_retrieve_convert, [], {}, 'call_options')

            loom.add_function(self.put_lis_retrieve_convert, [], {}, 'put_options')

            output = loom.execute()

            call_lis = output['call_options']['output']

            put_lis = output['put_options']['output']

            all_options.append(call_lis)

            all_options.append(put_lis)

            return all_options

        except:

            return ['Not found']



    def call_lis_retrieve_convert(self):

        call_lis = []

        option = Call(self.ticker, d=self.expiration.day, m=self.expiration.month, y=self.expiration.year)

        for strike in option.strikes:

            call_lis.append(Options.option_to_json(

                Call(self.ticker, d=self.expiration.day, m=self.expiration.month, y=self.expiration.year,

                     strike=strike)))

        return call_lis



    def put_lis_retrieve_convert(self):

        put_lis = []

        option = Put(self.ticker, d=self.expiration.day, m=self.expiration.month, y=self.expiration.year)

        for strike in option.strikes:

            put_lis.append(Options.option_to_json(

                Put(self.ticker, d=self.expiration.day, m=self.expiration.month, y=self.expiration.year,

                    strike=strike)))

        return put_lis



    def get_data(self):

        return self.data



    @staticmethod

    def option_to_json(option):

        return {

            "strike": option.strike,

            "expiration": option.expiration,

            "ticker": option.ticker,

            "bid": option.bid,

            "ask": option.ask,

            "price": option.price,

            "id": option.id,

            "exchange": option.exchange,

            "cp": option.cp,

            "volume": option.volume,

            "open_interest": option.open_interest,

            "Option_type": option.Option_type,

            "code": option.code,

            "implied_volatility": option.implied_volatility(),

            "delta": option.delta(),

            "gamma": option.gamma(),

            "vega": option.vega(),

            "theta": option.theta(),

            "rho": option.rho()

        }

class StrategyData:



    """

    Each strategy is to return a max profit under the given condition and breakeven point

    also the strategy of how to get one along with all the options required.

    """



    def __init__(self, ticker, view, high, low, expiration):

        """

        :param ticker: String with ticker

        :param view: view of the deal

        :param high: defines the upper limit of the view

        :param low: defines the lower limit of the view

        :param expiration: date object containing the expiration

        """

        self.ticker = ticker

        self.view = view

        self.low = low

        self.high = high

        self.expiration = expiration

        print('Getting Options...')

        self.options = Options(ticker, expiration).get_data()

        print('Calculating volatility...')

        self.volatility = Volatility(ticker, expiration)

from pexecute.process import ProcessLoom



"""

view = {

    1: "Above",

    2: "Between",

    3: "Below",

}



This class uses function for the buy/sell of call/buy option and with graph analysis gives the required values

"""





class SimpleBuySellOptions:

    def __init__(self, strategy):

        self.strategy = strategy

        self.call_options = self.strategy.options[0]

        self.put_options = self.strategy.options[1]



    @staticmethod

    def get_info():

        return {

            'title': 'Buy Sell options',

            'description': 'This strategy uses the volatility to evaluate if the option is over or under priced and makes decision accordingly'

        }



    @staticmethod

    def call_buy_graph(option):

        price = option['price']

        strike = option['strike']



        def func(x):

            if x < strike:

                return -price

            return x - (price + strike)



        return func



    @staticmethod

    def call_sell_graph(option):

        price = option['price']

        strike = option['strike']



        def func(x):

            if x < strike:

                return price

            return -x + (price + strike)



        return func



    @staticmethod

    def put_buy_graph(option):

        price = option['price']

        strike = option['strike']



        def func(x):

            if x < strike:

                return -x + (strike - price)

            return price



        return func



    @staticmethod

    def put_sell_graph(option):

        price = option['price']

        strike = option['strike']



        def func(x):

            if x < strike:

                return x - (strike - price)

            return price



        return func



    def choose_buy_sell_option(self, option):

        """

        numpy.bool_ works this way to check with not True to give the required value

        :param option:

        :return:

        """

        if self.strategy.volatility.is_option_overpriced(option) is not True:

            return 'Sell'

        return 'Buy'



    def get_best_strategies(self):

        """

        This is a wrapper function for this class which processes and returns the data to other classes

        :return:

        """

        data = self._view_selection()

        return self._sort_trim_data(data)



    def _sort_trim_data(self, data):

        """

        Sorts the data with volume and gets the top 3 results.

        :param data:

        :return:

        """

        return_top = 3

        call_options = data[0]

        put_options = data[1]



        def find_max_volume_profit_loss_ratio(data):

            max_volume = 0

            max_ratio = 0

            for option in data:

                max_volume = max(max_volume, option['option']['volume'])

                max_ratio = max(max_ratio,

                                (option['max_profit'] if option['max_profit'] != 'Infinite' else 1) /

                                (option['max_loss'] if option['max_loss'] != 'Infinite' else 1))

            return max_volume, max_ratio



        call_option_max_volume, call_option_max_ratio = find_max_volume_profit_loss_ratio(data[0])

        put_option_max_volume, put_option_max_ratio = find_max_volume_profit_loss_ratio(data[1])



        def sort_method(option):

            temp_max_profit = option['max_profit']

            if option['max_profit'] == 'Infinite':

                temp_max_profit = 1

            temp_max_loss = option['max_loss']

            if option['max_loss'] == 'Infinite':

                temp_max_loss = 1



            profit_loss_ratio = ((temp_max_profit / temp_max_loss) / (call_option_max_ratio if option['option']['Option_type'] == 'Call' else put_option_max_ratio))

            volume_ratio = (option['option']['volume'] / (

                call_option_max_volume if option['option']['Option_type'] == 'Call' else put_option_max_volume))

            

            return profit_loss_ratio + volume_ratio



        sorted_trimmed_call_options = sorted(call_options, key=lambda option: sort_method(option), reverse=True)[0:return_top]

        sorted_trimmed_put_options = sorted(put_options, key=lambda option: sort_method(option), reverse=True)[0:return_top]

        return [sorted_trimmed_call_options, sorted_trimmed_put_options]





    def _view_selection(self):

        to_return = None

        if self.strategy.view == 1:

            to_return = self._view1(self.strategy.high)

        elif self.strategy.view == 2:

            to_return = self._view2(self.strategy.high, self.strategy.low)

        elif self.strategy.view == 3:

            to_return = self._view3(self.strategy.low)



        return to_return



    def _view1(self, high):

        """

        when the price stays above given value

        :return:

        """

        loom = ProcessLoom(max_runner_cap=10)

        loom.add_function(self._call_lis_view1, [high], {}, 'call_options')

        loom.add_function(self._put_lis_view1, [high], {}, 'put_options')

        output = loom.execute()

        call_lis = output['call_options']['output']

        put_lis = output['put_options']['output']

        lis = []

        lis.append(call_lis)

        lis.append(put_lis)

        return lis



    def _view2(self, high, low):

        """

        price stays in between high and low values

        :return:

        """

        loom = ProcessLoom(max_runner_cap=10)

        loom.add_function(self._call_lis_view2, [high, low], {}, 'call_options')

        loom.add_function(self._put_lis_view2, [high, low], {}, 'put_options')

        output = loom.execute()

        call_lis = output['call_options']['output']

        put_lis = output['put_options']['output']

        lis = []

        lis.append(call_lis)

        lis.append(put_lis)

        return lis



    def _view3(self, low):

        """

        price always stays below given value

        :return:

        """

        loom = ProcessLoom(max_runner_cap=10)

        loom.add_function(self._call_lis_view3, [low], {}, 'call_options')

        loom.add_function(self._put_lis_view3, [low], {}, 'put_options')

        output = loom.execute()

        call_lis = output['call_options']['output']

        put_lis = output['put_options']['output']

        lis = []

        lis.append(call_lis)

        lis.append(put_lis)

        return lis



    def _put_lis_view1(self, high):

        put_lis = []

        for option in self.put_options:

            put_sell_graph = self.put_sell_graph(option)

            put_buy_graph = self.put_buy_graph(option)



            if self.choose_buy_sell_option(option) == 'Sell':

                trade = 'Sell'

                break_even_point = option['strike'] - option['price']

                max_profit = option['price']

                max_loss = -put_sell_graph(high) if put_sell_graph(high) < 0 else 0



            else:

                trade = 'Buy'

                break_even_point = option['strike'] - option['price']

                max_profit = put_buy_graph(high) if put_buy_graph(high) > 0 else 0

                max_loss = option['price']



            put_lis.append({

                'trade': trade,

                'max_profit': max_profit,

                'max_loss': max_loss,

                'break_even_point': break_even_point,

                'option': option

            })

        return put_lis



    def _call_lis_view1(self, high):

        call_lis = []

        for option in self.call_options:

            call_sell_graph = self.call_sell_graph(option)

            call_buy_graph = self.call_buy_graph(option)



            if self.choose_buy_sell_option(option) == 'Sell':

                trade = 'Sell'

                max_profit = call_sell_graph(high) if call_sell_graph(high) > 0 else 0

                break_even_point = option['strike'] + option['price']

                max_loss = 'Infinite'

            else:

                trade = 'Buy'

                max_profit = 'Infinite'

                max_loss = -call_buy_graph(high) if call_buy_graph(high) < 0 else 0

                break_even_point = option['strike'] + option['price']



            call_lis.append({

                'trade': trade,

                'max_profit': max_profit,

                'max_loss': max_loss,

                'break_even_point': break_even_point,

                'option': option

            })

        return call_lis



    def _call_lis_view2(self, high, low):

        call_lis = []

        for option in self.call_options:

            call_sell_graph = self.call_sell_graph(option)

            call_buy_graph = self.call_buy_graph(option)



            if self.choose_buy_sell_option(option) == 'Sell':

                trade = 'Sell'

                break_even_point = option['strike']

                max_profit = call_sell_graph(low) if call_sell_graph(low) > 0 else 0

                max_loss = -call_sell_graph(high) if call_sell_graph(high) < 0 else 0

            else:

                trade = 'Buy'

                break_even_point = option['strike']

                max_profit = call_buy_graph(high) if call_buy_graph(high) > 0 else 0

                max_loss = -call_buy_graph(low) if call_buy_graph(low) < 0 else 0

            call_lis.append({

                'trade': trade,

                'max_profit': max_profit,

                'max_loss': max_loss,

                'break_even_point': break_even_point,

                'option': option

            })

        return call_lis



    def _put_lis_view2(self, high, low):

        put_lis = []

        for option in self.put_options:

            put_sell_graph = self.put_sell_graph(option)

            put_buy_graph = self.put_buy_graph(option)

            if self.choose_buy_sell_option(option) == 'Sell':

                trade = 'Sell'

                break_even_point = option['strike'] - option['price']

                max_profit = put_sell_graph(high) if put_sell_graph(high) > 0 else 0

                max_loss = -put_sell_graph(low) if put_sell_graph(low) < 0 else 0

            else:

                trade = 'Buy'

                break_even_point = option['strike'] - option['price']

                max_profit = put_buy_graph(low) if put_buy_graph(low) > 0 else 0

                max_loss = -put_buy_graph(high) if put_buy_graph(high) < 0 else 0



        put_lis.append({

            'trade': trade,

            'max_profit': max_profit,

            'max_loss': max_loss,

            'break_even_point': break_even_point,

            'option': option

        })

        return put_lis



    def _call_lis_view3(self, low):

        call_lis = []

        for option in self.call_options:

            call_sell_graph = self.call_sell_graph(option)

            call_buy_graph = self.call_buy_graph(option)



            if self.choose_buy_sell_option(option) == 'Sell':

                trade = 'Sell'

                break_even_point = option['strike']

                max_profit = call_sell_graph(0)

                max_loss = -call_sell_graph(low) if call_sell_graph(low) < 0 else 0

            else:

                trade = 'Buy'

                break_even_point = option['strike']

                max_profit = call_buy_graph(low) if call_buy_graph(low) > 0 else 0

                max_loss = call_buy_graph(0)

            call_lis.append({

                'trade': trade,

                'max_profit': max_profit,

                'max_loss': max_loss,

                'break_even_point': break_even_point,

                'option': option

            })

        return call_lis



    def _put_lis_view3(self, low):

        put_lis = []

        for option in self.put_options:

            put_sell_graph = self.put_sell_graph(option)

            put_buy_graph = self.put_buy_graph(option)

            if self.choose_buy_sell_option(option) == 'Sell':

                trade = 'Sell'

                break_even_point = option['strike'] - option['price']

                max_profit = put_sell_graph(low) if put_sell_graph(low) > 0 else 0

                max_loss = put_sell_graph(0)

            else:

                trade = 'Buy'

                break_even_point = option['strike'] - option['price']

                max_profit = put_buy_graph(0)

                max_loss = -put_buy_graph(low) if put_buy_graph(low) < 0 else 0



        put_lis.append({

            'trade': trade,

            'max_profit': max_profit,

            'max_loss': max_loss,

            'break_even_point': break_even_point,

            'option': option

        })

        return put_lis

from wallstreet import Stock, Call

from datetime import datetime





def _convert_datetime(date):

    return datetime.strptime(date, '%d-%m-%Y').date()





class StrategyClassifier:

    """

    Class is used to get the strategies and sort them also acts as a wrapper class for all the strategy related methods in server.py

    """



    def __init__(self, ticker, view, low, high, expiration):

        self.ticker = ticker

        self.view = view

        self.low = low

        self.high = high

        self.expiration = _convert_datetime(expiration)

        self.strategy_data = StrategyData(ticker, view, low, high, self.expiration)



    @staticmethod

    def get_views():

        return {

            1: "Above",

            2: "Between",

            3: "Below",

        }



    @staticmethod

    def is_ticker_valid(ticker):

        try:

            Stock(ticker)

            return True

        except:

            return False



    @staticmethod

    def get_stock_current_value_expirations(ticker):

        if not StrategyClassifier.is_ticker_valid(ticker):

            return None

        else:

            option = Call(ticker)

            return {

                'price': option.price,

                'expirations': option.expirations

            }



    def get_strategies(self):

        simple_buy_sell_option = SimpleBuySellOptions(self.strategy_data)

        return [

            {

                'title': SimpleBuySellOptions.get_info()['title'],

                'description': SimpleBuySellOptions.get_info()['description'],

                'data': simple_buy_sell_option.get_best_strategies()

            }

        ]

from datetime import date

strategy_classifier = StrategyClassifier('AAPL', 1, 0, 300, "19-06-2020")
import json

s = strategy_classifier.get_strategies()

s_json = json.dumps(strategy_classifier.get_strategies())
s[0]