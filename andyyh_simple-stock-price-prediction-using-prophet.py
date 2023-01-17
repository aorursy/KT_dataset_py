# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Setting up plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl

import plotly.offline as py

init_notebook_mode(connected=True)

import plotly.graph_objects as go 

import seaborn as sns

import plotly

import plotly.express as px

from fbprophet.plot import plot_plotly

from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot

import sklearn

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, RidgeCV

from sklearn.svm import LinearSVR

import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/6000-nasdaq-stocks-historical-daily-prices/nasdaq_historical_prices_daily/nasdaq_historical_prices_daily.csv', delimiter=',')

df.head()
ticker = 'AAPL'

df2 = df.loc[df['ticker'] == ticker]

df3 = df2[:365]

df3.shape
fig = go.Figure()

fig.update_layout(template='seaborn')

fig.add_trace(go.Scatter(x=df3['date'], 

                         y=df3['close'],

                         mode='lines+markers',

                         name=ticker,

                         line=dict(color='Blue', width=2)))



fig.show()
X_df = df3.drop(columns='close')

y = df3['close'].values
encoder = OneHotEncoder()

X = encoder.fit_transform(X_df.values)

X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=25)
model = LinearRegression()

model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)

test_score = model.score(X_test, y_test)

print("Train Score:", train_score)

print("Test Score:", test_score)
training = df3.rename(columns={'date': 'ds', 'close': 'y'})
def compute_mse(model, X, y_true, name):

    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)

    print(f'Mean Squared Error for {name}: {mse}')

    

compute_mse(model, X_train, y_train, 'training set')

compute_mse(model, X_test, y_test, 'test set')
m = Prophet()

m.fit(training)

future = m.make_future_dataframe(periods=365)

forecast = m.predict(future)
fig = plot_plotly(m, forecast)

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Predictions for ' + ticker + ' stock price',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig