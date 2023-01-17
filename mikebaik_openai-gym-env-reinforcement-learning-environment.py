# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/Data/Stocks/")[:10])
stock_path = '../input/Data/Stocks/'
# Any results you write to the current directory are saved as output.
import itertools
import functools
import matplotlib.pyplot as plt
import collections
import six
import gym
import datetime
import gym.spaces

def iterable(arg):
    return (isinstance(arg, collections.Iterable) and not
            isinstance(arg, six.string_types))
class TickerContinuous:
    # Don't delete num_actions just yet, need to go fix all others..
    #   Especially when constructing in Engine
    def __init__(self, ticker, start_date, num_days_iter,
                 today=None, num_actions=3, test=False,
                 action_space_min=-1.0, action_space_max=1.0):
        self.ticker = ticker
        self.start_date = start_date
        self.num_days_iter = num_days_iter
        self.df, self.dates = self._load_df(test)
        self.action_space = gym.spaces.Box(action_space_min, action_space_max,
                                           (1, ), dtype=np.float32)
        self.today = 0 if today is None else today
        self._data_valid()
        self.current_position = self.accumulated_pnl = 0.0

    def _load_df(self, test):
        if test:
            ticker_data = self._load_test_df()
        else:
            ticker_data = pd.read_csv(stock_path+f'{self.ticker}.us.txt')
            # print(ticker_data.columns)
            # ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']
            ticker_data.rename(lambda x: str.lower(x), axis=1, inplace=True)
            ticker_data.drop('openint', axis=1, inplace=True)
            ticker_data = ticker_data[ticker_data['date'] >= self.start_date]
            
        ticker_data.reset_index(inplace=True)
        # This is really cheating but...
        dates_series = ticker_data['date']
        ticker_data.drop('date', axis=1, inplace=True)
        # This part should become a function eventually
        ticker_data_delta = ticker_data.pct_change()
        add_str_delta = lambda x: x + '_delta'
        ticker_data_delta.rename(add_str_delta, axis='columns', inplace=True)
        ticker_data_delta.iloc[0, :] = 0.0

        zeros = pd.DataFrame(np.zeros((len(ticker_data), 2)),
                             columns=['position', 'pnl'])

        # It's probably better to transpose, then let columns be dates, but wtf...
        df = pd.concat([ticker_data, ticker_data_delta, zeros], axis=1)
        df.drop(['index', 'index_delta'], axis=1, inplace=True)

        return df, dates_series

    def _load_test_df(self):
        date_col = [datetime.date.today() + datetime.timedelta(days=i)
                    for i in range(self.num_days_iter)]
        aranged_values = [np.repeat(i, 6) for i in range(1, self.num_days_iter+1)]
        temp_df = pd.DataFrame(aranged_values,
                               columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        temp_df.iloc[:, 0] = date_col
        return temp_df

    def _data_valid(self):
        assert len(self.df) >= self.num_days_iter, \
                f'DataFrame shape: {self.df.shape}, num_days_iter: {self.num_days_iter}'
        assert len(self.df) == len(self.dates), \
                f'df.shape: {self.df.shape}, dates.shape:{self.dates.shape}'

    def get_state(self, delta_t=0):
        today_market_data_position = np.array(self.df.iloc[self.today+delta_t, -7:-2])
        today_market_data_position[-1] = self.current_position
        return today_market_data_position

    # 1. Reward is tricky
    # 2. Should invalid action be penalized?
    def step(self, action):
        if not self.done():
            # Record pnl
            # This implementation of reward is such a hogwash!!
            #     but recall, Deepmind's DQN solution does something similar...
            #     assigning credit is always hard...
            # Pandas complain here, "A value is trying to be set on a copy of a slice from a DataFrame"
            #     but the suggested solution is actually misleading... so leaving it as is
            pd.set_option('mode.chained_assignment', None)
            self.df.pnl[self.today] = reward = 0.0 if self.today == 0 else \
                                               self.current_position * self.df.close_delta[self.today]

            # Think about accumulating the scores...
            self.accumulated_pnl += reward
            self.df.position[self.today] = self.current_position = action
            self.today += 1

            return reward, False
        else:
            self.current_position = 0.0
            return 0.0, True

    def valid_action(self, action):
        return self.action_space.low <= action <= self.action_space.high

    def reset(self):
        self.today = 0
        self.df.position = self.df.pnl = 0.0
        self.current_position = self.accumulated_pnl = 0.0

    # NOT THE MOST EFFICIENT...
    def done(self):
        return self.today > self.num_days_iter

    def render(self, axis):
        # market_data, position = self.get_state()
        # axis[0].scatter(self.today, self.df.pnl[self.today-1])
        axis[0].set_ylabel(f'Daily price: {self.ticker}')
        axis[0].set_xlabel('Time step')
        axis[0].plot(np.arange(self.today), self.df.close[:self.today])
        # axis[1].scatter(self.today, position)
        # axis[2].scatter(self.today, self.accumulated_pnl)
        axis[1].set_ylabel(f'Daily return from Agent')
        axis[1].set_xlabel('Time step')
        axis[1].scatter(self.today, self.accumulated_pnl)
        plt.pause(0.0001)
class EngineContinuous:
    def __init__(self, tickers, start_date, num_days_iter,
                 today=None, seed=None, num_action_space=3,
                 render=False, *args, **kwargs):
        if seed: np.random.seed(seed)
        if not iterable(tickers): tickers = [tickers]

        self.tickers = self._get_tickers(tickers, start_date, num_days_iter,
                                         today, num_action_space, *args, **kwargs)
        self.reset_game()

        if render:
            # Somehow ax_list should be grouped in two always...
            # Or is there another way of getting one axis per row and then add?
            fig_height = 3 * len(self.tickers)
            self.fig, self.ax_list = plt.subplots(len(tickers), 2, figsize=(10, fig_height))

    def reset_game(self):
        list(map(lambda ticker: ticker.reset(), self.tickers))

    def _get_tickers(self, tickers, start_date, num_days_iter,
                     today, num_action_space, *args, **kwargs):
        return [TickerContinuous(ticker, start_date, num_days_iter, today, num_action_space, *args, **kwargs)
                for ticker in tickers]

    def _render(self, render):
        if render:
            if len(self.tickers) == 1:
                self.tickers[0].render(self.ax_list)
            else:
                for axis, ticker in zip(self.ax_list, self.tickers):
                    ticker.render(axis)

    def get_state(self, delta_t=0):
        # Note: np.arary(...) could also be used
        return list(map(lambda ticker: ticker.get_state(delta_t), self.tickers))

    def moves_available(self):
        raise NotImplementedError

    def step(self, actions):
        if not iterable(actions): actions = [actions]
        assert len(self.tickers) == len(actions), f'{len(self.tickers)}, {len(actions)}'

        rewards, dones = zip(*(itertools.starmap(lambda ticker, action: ticker.step(action),
                                                 zip(self.tickers, actions))))

        # This is somewhat misleading
        score = functools.reduce(lambda x, y: x + y, rewards, 0.0)
        done = functools.reduce(lambda x, y: x | y, dones, False)

        return score, done

    def render(self, render=False):
        # This is possibly unnecessary b/c of changes
        self._render(render)

    def __repr__(self):
        tickers = [f'ticker_{i}: {ticker.ticker}, ' for i, ticker in enumerate(self.tickers)]
        return str(tickers)

    def _data_valid(self):
        raise NotImplementedError

class PortfolioContinuous(EngineContinuous):
    def __init__(self, tickers, start_date, num_days_iter,
                 today=None, seed=None, render=False,
                 action_space_min=0.0, action_space_max=1.0):
        num_action_space = len(tickers)
        super().__init__(tickers, start_date, num_days_iter,
                         today, seed, num_action_space, render,
                         action_space_min=action_space_min,
                         action_space_max=action_space_max)
        self.action_space = gym.spaces.Box(action_space_min, action_space_max,
                                           (num_action_space, ), np.float32)

    def step(self, actions):
        return super(PortfolioContinuous, self).step(actions)
import gym.spaces as spaces
class StockExchangeContinuous(gym.Env):
    metadata = {'render.modes': ['human']}

    # Keep tickers in a list or an iterable...
    tickers = ['aapl', 'amd', 'msft', 'intc', 'd', 'sbux', 'atvi',
               'ibm', 'ual', 'vrsn', 't', 'mcd', 'vz']
    start_date = '2013-09-15'
    num_days_to_iterate = 100
    num_days_in_state = 20
    num_action_space = len(tickers)
    # no_action_index is truly no_action only if it's not a Portfolio
    no_action_index = num_action_space//2
    today = 0
    render = False
    # set to None when not using Portfolio
    action_space_min = -1.0
    action_space_max = 1.0
    # For each ticker state: ohlc
    num_state_per_ticker = 4

    def __init__(self, seed=None):

        # Could manually throw in options eventually...
        self.portfolio = self.num_action_space > 1
        self._seed = seed

        if self.portfolio:
            assert self.action_space_min is not None
            assert self.action_space_max is not None
            self.env = PortfolioContinuous(self.tickers, self.start_date,
                                           self.num_days_to_iterate,
                                           self.today, seed, render=self.render,
                                           action_space_min=self.action_space_min,
                                           action_space_max=self.action_space_max)
        else:
            assert self.num_action_space % 2 != 0, 'NUM_ACTION_SPACE MUST BE ODD TO HAVE NO ACTION INDEX'
            self.env = EngineContinuous(self.tickers, self.start_date,
                                        self.num_days_to_iterate,
                                        self.today, seed,
                                        num_action_space=self.num_action_space,
                                        render=self.render)

        self.action_space = gym.spaces.Box(self.action_space_min, self.action_space_max,
                                       (self.num_action_space, ), np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0,
                                            (self.num_days_in_state,
                                             self.num_action_space * self.num_state_per_ticker),
                                            dtype=np.float32)
        self.state = self.get_running_state()
        self.reset()

    def step(self, actions):
        # I can fix Engine to return state from `self.env.step(action)`
        reward, ended = self.env.step(actions)
        self.state = self.add_new_state(self.env.get_state())
        return self.state, reward, ended, {'score': reward}

    def reset(self):
        self.env.reset_game()
        self._initialize_state()
        return self.state

    def render(self, mode='human', render=False):
        self.env.render(render)

    def _initialize_state(self):
        for _ in range(self.num_days_in_state - 1):
            if self.portfolio:
                zero_action = [0.0] * self.num_action_space
                next_state, reward, done, _ = self.step(zero_action)
            else:
                next_state, reward, done, _ = self.step([self.no_action_index] * self.num_action_space)
                assert reward == 0.0, f'Reward is somehow {reward}'

    def __repr__(self):
        return repr(self.env)

    def get_running_state(self):
        return np.zeros((self.num_days_in_state, self.num_state_per_ticker * self.num_action_space))

    def add_new_state(self, new_states_to_add):
        assert isinstance(new_states_to_add, list), type(new_states_to_add)
        # Disregarding the last elem in each state because it's the holdings...
        # Maybe just get rid of that altogether?
        new_states = np.array([state[:-1].tolist() for state in new_states_to_add]).flatten()

        running_state_orig = self.state
        running_state = pd.DataFrame(running_state_orig).shift(-1)

        # Assign new price to index == last_elem - 1
        running_state.iloc[-1] = new_states.squeeze()

        # Deprecated...
        # running_state.iloc[-2] = new_state_to_add.item(0)
        # Assign new position to index == last_elem
        # running_state.iloc[-1] = new_state_to_add.item(1)

        assert len(running_state_orig) == len(running_state)
        return np.array(running_state)

env = StockExchangeContinuous()
class RandomAgent():
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()
agent = RandomAgent(env.action_space)
episode_count = 2
rewards = []
done = False

for _ in range(episode_count):
    ob = env.reset()
    while True:
        action = agent.act()
        ob, reward, done, _ = env.step(action)
        rewards += [reward]
        if done:
            break
np.mean(rewards), np.std(rewards)
