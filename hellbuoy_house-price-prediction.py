# import all libraries and dependencies for dataframe

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta



# import all libraries and dependencies for data visualization

pd.options.display.float_format='{:.4f}'.format

plt.rcParams['figure.figsize'] = [8,8]

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', -1) 

sns.set(style='darkgrid')

import matplotlib.ticker as ticker

import matplotlib.ticker as plticker
# Reading the Uber file



path = '../input/house-prices-advanced-regression-techniques/'

file = path + 'train.csv'

train = pd.read_csv(file)
train.shape
# Describe the data



train.describe()
# Data info



train.info()
# Data Shape



train.shape
train.dtypes
# Segregation of Numerical and Categorical Variables/Columns



cat_col = train.select_dtypes(include=['object']).columns

num_col = train.select_dtypes(exclude=['object']).columns

df_cat = train[cat_col]

df_num = train[num_col]

# PairPlot to understand the relationship between Numerical columns and Target variable



#sns.pairplot(df_num)
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (20, 20))

sns.heatmap(train.corr(), cmap="RdYlGn")

plt.show()
train = train.drop('Id',axis=1)
# Calculating the Missing Values % contribution in DF



df_null = train.isna().mean().round(4) *100

df_null
# Plot for Attributes where NaN values more than 30%



plt.rcParams['figure.figsize'] = [15,8]

df_na_col = train.isnull().sum()

df_na_col = df_na_col[df_na_col.values >(0.2*len(train))]

df_na_col.sort_values().plot(kind='bar',stacked=True, colormap = 'Set1')

plt.title('Columns with NaN values more than 30%', fontweight = 'bold')

plt.xlabel("Columns", fontweight = 'bold')

plt.ylabel("NaN Count", fontweight = 'bold')
# Dropping the Columns where all rows are Unique



uni = train.nunique()

uni = uni[uni.values == 1].copy()

train.drop(labels = list(uni.index), axis =1, inplace=True)