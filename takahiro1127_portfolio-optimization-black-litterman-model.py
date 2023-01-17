! pip install pyfolio

! pip install PyPortfolioOpt
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from pypfopt import risk_models as RiskModels

import datetime

from pypfopt import black_litterman

from pypfopt.black_litterman import BlackLittermanModel

from pypfopt.efficient_frontier import EfficientFrontier

%matplotlib inline

# portfolio optimizationのライブラリを使用

# https://pyportfolioopt.readthedocs.io/en/latest/
prices = pd.read_csv("/kaggle/input/group3data/his_data.csv", names=("Date", "JPY", "NZD", "AUD", "EUR", "GBP", "HKD", "BRL", "DKK", "INR", "CAD", "CHF")).drop(0)

prices['Date'] = prices['Date'].astype('str')

prices['Date'] = prices["Date"].str[:4] + "-" + prices["Date"].str[4:6] + "-" + prices["Date"].str[6:8]

prices['Date'] = pd.to_datetime(prices['Date'])

prices = prices.set_index('Date')

currencies = ["JPY", "NZD", "AUD", "EUR", "GBP", "HKD", "BRL", "DKK", "INR", "CAD", "CHF"]

for currency in currencies:

    prices[currency] = prices[currency].astype(float)
shrunk_covariance = RiskModels.CovarianceShrinkage(prices)

shrunk_covariance = shrunk_covariance.shrunk_covariance()
weight_set = []

P = np.array([

        [-0.3,-0.1,-0.2,-0.2,-0.1,-0.1,1, 0, 0, 0, 0],# BRL up 3%

        [1,0,0,0,0,0,0,0,0,0,0]]) # JPY up 1%

Q = np.array([[0.0003],[0.001]]) # 2-vector

delta = black_litterman.market_implied_risk_aversion(prices["JPY"])

Omega = BlackLittermanModel.default_omega(cov_matrix = shrunk_covariance, P = P, tau = 0.05)

bl = BlackLittermanModel(shrunk_covariance, P = P, Q = Q, omega = Omega)

rets = bl.bl_returns()

bl.bl_weights(delta)

weight = bl.clean_weights()
df = pd.DataFrame(weight,

                  columns=currencies,

                  index=['Equilibrium Weights'])

df.T.plot(kind='bar')