import os

import numpy as np

import pandas as pd

import xgboost as xgb

import seaborn as sns

from xgboost import DMatrix

import matplotlib.pyplot as plt

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import confusion_matrix, recall_score

import lightgbm as lgb

from sklearn.metrics import accuracy_score

from tqdm import tqdm

import plotly as py

import plotly.io as pio

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from warnings import simplefilter
# current price of underlying asset

current_price = 100

# array of pair of probability and tommorow price

# lower_price must be lower than current price, higher_price must be higher than current price, 

patterns = {'lower_price' : {'probability': 0.5, 'tomorrow_rate': 0.9}, 'higher_price' : {'probability': 0.5, 'tomorrow_rate': 1.2}}

# ten year risk free rate

ten_year_risk_free_rate = 0.0081

# one day risk free rate

one_day_risk_free_rate = 1.0081 ** (1/3650)

# calc probability based on risk free rate and tomorrow rate

q = (one_day_risk_free_rate - patterns['lower_price']['tomorrow_rate']) / (patterns['higher_price']['tomorrow_rate'] - patterns['lower_price']['tomorrow_rate'])
def option_premium_depends_on_tomorrow_rate_and_strike_price(strike_price, pattern):

    return max(pattern['tomorrow_rate'] * current_price - strike_price, 0)



def option_premium_based_on_probability(strike_price, probability = q):

    return (probability * option_premium_depends_on_tomorrow_rate_and_strike_price(strike_price, patterns['higher_price']) + (1 - probability) * option_premium_depends_on_tomorrow_rate_and_strike_price(strike_price, patterns['lower_price'])) / one_day_risk_free_rate
def doller_to_have_stock(strike_price):

    return (option_premium_depends_on_tomorrow_rate_and_strike_price(strike_price, patterns['higher_price']) - option_premium_depends_on_tomorrow_rate_and_strike_price(strike_price, patterns['lower_price'])) / (patterns['higher_price']['tomorrow_rate'] - patterns['lower_price']['tomorrow_rate'])



def doller_to_have_risk_free(strike_price):

    return (patterns['higher_price']['tomorrow_rate'] * option_premium_depends_on_tomorrow_rate_and_strike_price(strike_price, patterns['lower_price']) - patterns['lower_price']['tomorrow_rate'] * option_premium_depends_on_tomorrow_rate_and_strike_price(strike_price, patterns['higher_price'])) / (one_day_risk_free_rate * (patterns['higher_price']['tomorrow_rate'] - patterns['lower_price']['tomorrow_rate']))
results = pd.DataFrame(columns = ['strike_price', 'option_premium', 'option_premium_equal_probability', 'stock_doller_in_portfolio', 'risk_free_doller_in_portfolio'])

strike_prices = []

option_premiums = []

stock_dollers_in_portfolio = []

option_premiums_equal_probability = []

risk_free_dollers_in_portfolio = []

for strike_price in range(max(current_price - 1000, 0), current_price * 2):

    strike_prices.append(strike_price)

    option_premiums.append(option_premium_based_on_probability(strike_price, q))

    option_premiums_equal_probability.append(option_premium_based_on_probability(strike_price, 0.5))

    stock_dollers_in_portfolio.append(doller_to_have_stock(strike_price))

    risk_free_dollers_in_portfolio.append(doller_to_have_risk_free(strike_price))   

results['strike_price'] = strike_prices

results['option_premium'] = option_premiums

results['option_premium_equal_probability'] = option_premiums_equal_probability

results['stock_doller_in_portfolio'] = stock_dollers_in_portfolio

results['risk_free_doller_in_portfolio'] = risk_free_dollers_in_portfolio

results = results.set_index('strike_price')

results.plot()
results[105:106] * 1000