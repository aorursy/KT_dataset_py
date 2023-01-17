import pandas as pd

import numpy as np

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
raw_data = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")

raw_data.info()
raw_data.head()
suicides_gdp = (raw_data.groupby(['country', 'year'],as_index = False)

.agg({'suicides_no':'sum','gdp_per_capita ($)': 'mean'}))
country = suicides_gdp['country'].unique()
len(country)
suicides_gdp.head()
suicides_gdp.info()
x = suicides_gdp['gdp_per_capita ($)']

y = suicides_gdp['suicides_no']

plt.xlabel("gdp per capita ($)")

plt.ylabel("number of suicides")

plt.scatter(x, y)

plt.show()
x.corr(y)