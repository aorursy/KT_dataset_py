import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/master.csv')
data.head()
data.shape
data.info()
data.dtypes
data.dropna()
data.shape
data.columns.values
data.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population',

       'suicidesper100kpop', 'country-year', 'HDI for year',

       'gdp_for_year_dollars', 'gdp_per_capita_dollars', 'generation']
data['gdp_for_year_dollars'] = data['gdp_for_year_dollars'].str.replace(',','').astype(int)

data.info()
data.isnull().sum().sort_values(ascending=False)
data_n = data.drop(['HDI for year', 'country-year'], axis=1)
data_n.head(5)
data_n.describe()
data_n.describe(include=['O'])
data_n[['sex','suicides_no']].groupby(['sex']).mean().plot(kind='bar')