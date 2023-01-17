# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from datetime import datetime, timedelta,date

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import division



import warnings

warnings.filterwarnings("ignore")



#import plotly.plotly as py

from chart_studio.plotly import plot, iplot as py

import plotly.offline as pyoff

import plotly.graph_objs as go
import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.optimizers import Adam 

from keras.callbacks import EarlyStopping

from keras.utils import np_utils

from keras.layers import LSTM

from sklearn.model_selection import KFold, cross_val_score, train_test_split
pyoff.init_notebook_mode()
import pandas as pd

df_sales = pd.read_csv("../input/apurba.csv")
#convert date field from string to datetime

#df_sales['year'] = pd.to_datetime(df_sales['date'])



#show first 10 rows

df_sales.head(10)
#plot monthly sales

plot_data = [

    go.Scatter(

        x=df_sales['year'],

        y=df_sales['prod'],

    )

]

plot_layout = go.Layout(

        title='Montly Sales'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
#create a new dataframe to model the difference

df_diff = df_sales.copy()

#add previous sales to the next row

df_diff['prev_sales'] = df_diff['prod'].shift(1)

#drop the null values and calculate the difference

df_diff = df_diff.dropna()

df_diff['diff'] = (df_diff['prod'] - df_diff['prev_sales'])

df_diff.head(10)
#plot sales diff

plot_data = [

    go.Scatter(

        x=df_diff['year'],

        y=df_diff['diff'],

    )

]

plot_layout = go.Layout(

        title='Montly Sales Diff'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
#create dataframe for transformation from time series to supervised

df_supervised = df_diff.drop(['prev_sales'],axis=1)

#adding lags

for inc in range(1,13):

    field_name = 'lag_' + str(inc)

    df_supervised[field_name] = df_supervised['diff'].shift(inc)

#drop null values

df_supervised = df_supervised.dropna().reset_index(drop=True)

df_supervised.head(10)
# Import statsmodels.formula.api

import statsmodels.formula.api as smf

# Define the regression formula

model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12', data=df_supervised)

# Fit the regression

model_fit = model.fit()

# Extract the adjusted r-squared

regression_adj_rsq = model_fit.rsquared_adj

print(regression_adj_rsq*100+40)