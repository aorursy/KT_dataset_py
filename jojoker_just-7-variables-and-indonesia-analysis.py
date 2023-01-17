import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set()
raw_data = pd.read_csv('../input/world-happiness-report-2019/world happiness report 2019.csv')
raw_data.head()
raw_data.info()
country = raw_data['Country name'].unique()
count_country = len(country)
country
count_country
data = raw_data.iloc[:,0:6]

data = data.dropna(axis=0)
data.info()
data_indonesia = data.loc[raw_data['Country name'].isin(['Indonesia'])]
data_indonesia = data_indonesia.copy()
data_indonesia.info()
y = data['Log GDP per capita']

x = data[['Life Ladder',

       'Social support', 'Healthy life expectancy at birth']]

x1 = sm.add_constant(x)
result = sm.OLS(y,x1).fit()
result.summary()
gdp = data_indonesia['Log GDP per capita']

year = data_indonesia['Year']

plt.scatter(year,gdp)

plt.plot(year, gdp)

plt.show()
life_expectacy = data_indonesia['Healthy life expectancy at birth']

plt.scatter(year,life_expectacy)

plt.plot(year,life_expectacy)

plt.show()
social_support = data_indonesia['Social support']

plt.scatter(year,social_support)

plt.plot(year,social_support)

plt.show()
life_ladder = data_indonesia['Life Ladder']

plt.scatter(year,life_ladder)

plt.plot(year,life_ladder)

plt.show()