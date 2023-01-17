# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



# Any results you write to the current directory are saved as output.
company_name = "AAPL" # Type in a company name and just run the rest
filename= "/kaggle/input/sandp500/individual_stocks_5yr/individual_stocks_5yr/{}_data.csv".format(company_name)
df = pd.read_csv(filename)

df.head()
data = df[["date", "high"]]
data.columns = ["ds", "y"]

data.head()
import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format='retina'
data.plot()
from fbprophet import Prophet
m = Prophet()

m.fit(data)
future = m.make_future_dataframe(periods=730)

future.tail()
forecast = m.predict(future)
fig = m.plot(forecast)
fig2 = m.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()
fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)