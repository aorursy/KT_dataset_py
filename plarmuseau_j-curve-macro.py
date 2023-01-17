import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import model_selection, preprocessing

import xgboost as xgb

import datetime

#now = datetime.datetime.now()



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

macro = pd.read_csv('../input/macro.csv')

id_test = test.id
from matplotlib import pyplot

macropl=macro[['balance_trade']]

macropl.plot()

pyplot.show()
from matplotlib import pyplot

macropl2=macro[['cpi','ppi','usdrub','brent']]

macropl2.plot()

pyplot.show()
from matplotlib import pyplot

macropl=macro[['deposits_value']]

macropl2=macro[['mortgage_value','construction_value']]

macropl3=macro[['salary','invest_fixed_assets']]

macropl.plot()

macropl2.plot()

macropl3.plot()

pyplot.show()
from matplotlib import pyplot

macropl=macro[['deposits_rate','mortgage_rate']]

macropl2=macro[['grp_growth','gdp_annual_growth','salary_growth','deposits_growth']]

macropl.plot()

macropl2.plot()

pyplot.show()