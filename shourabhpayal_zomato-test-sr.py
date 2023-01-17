# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.metrics import confusion_matrix

#importing libraries

from sklearn import neighbors

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

import time

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")
df.head()
df.columns
df.describe()
df.isnull().sum()
df.dtypes
df["approx_cost(for two people)"].head()
df["approx_cost(for two people)"] = df["approx_cost(for two people)"].astype('str').apply(lambda x: x.replace(',', '')).astype(float)

df["approx_cost(for two people)"]
df['rate'] = df['rate'].astype('str').apply(lambda x: x.split('/')[0]).apply(lambda x: x.replace('NEW', str(np.nan))).apply(lambda x: x.replace('-', str(np.nan))).astype(float)

df['rate']
df.dropna(subset=['rate', 'approx_cost(for two people)'], inplace=True)

df.drop(['url', 'phone'], axis=1, inplace=True)

df.isnull().sum()
import seaborn as sb

plt.figure(1, figsize=(18, 7))

sb.set(style="whitegrid")

sb.countplot( x= 'rate', data=df)

plt.title('distribution of all rates')

plt.show()
import seaborn as sb

plt.figure(1, figsize=(18, 7))

sb.set(style="whitegrid")

ax = sb.countplot( x= 'approx_cost(for two people)', data=df)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.title('distribution of cost')

plt.tight_layout()

plt.show()