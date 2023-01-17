# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





!pip3 uninstall --yes fbprophet

!pip3 install fbprophet --no-cache-dir --no-binary :all:

# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv('../input/daily-minimum-temperatures-in-me.csv', error_bad_lines=False)

df.columns = ['ds','y']

df["y"] = df["y"].map(lambda x: x.lstrip('?'))

df["y"] = pd.to_numeric(df["y"])

df["ds"] = pd.to_datetime(df["ds"])

df.head()
from fbprophet import Prophet

m = Prophet()

m.fit(df)
future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)