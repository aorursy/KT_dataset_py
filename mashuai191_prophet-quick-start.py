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
# Python

import pandas as pd

from fbprophet import Prophet



df = pd.read_csv('/kaggle/input/example_wp_log_peyton_manning.csv')

df.head()
#df["ds"] = pd.to_datetime(df["ds"])

df["ds"]
m = Prophet()

m.fit(df)
future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

#forecast.tail()

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# must add following line otherwise plot will fail. 

# solution from: https://darektidwell.com/typeerror-float-argument-must-be-a-string-or-a-number-not-period-facebook-prophet-and-pandas/

pd.plotting.register_matplotlib_converters()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fig = plot_plotly(m, forecast)  # This returns a plotly Figure

py.iplot(fig)