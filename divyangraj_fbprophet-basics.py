# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

from fbprophet import Prophet
data = pd.read_csv("../input/nse-nairobi-securities-data-2012-june-2019/Agricultural/EGAD Historical Data.csv",parse_dates=["Date"])

data
data.shape
data.describe()
data.nunique()
import matplotlib.pyplot as plt

%matplotlib inline
data.hist("Price")
data.set_index('Date')['Price'].plot(figsize=(7, 5), linewidth=2.5, color='maroon')

plt.xlabel("Date", labelpad=15)

plt.ylabel("Price", labelpad=15)

plt.title("Moving_Price", y=1.02, fontsize=22);
data.set_index("Date")["Open"].plot(figsize=(7, 5), linewidth=2.5, color='maroon')

plt.xlabel("Date",labelpad = 15)

plt.ylabel("Open",labelpad = 15)

plt.title("Moving_Open_Price", y=1.02, fontsize=22);
data.set_index("Date")["High"].plot(figsize=(7, 5), linewidth=2.5, color='maroon')

plt.xlabel("Date",labelpad = 15)

plt.ylabel("High",labelpad = 15)

plt.title("Moving_High_Price", y=1.02, fontsize=22);
data.set_index("Date")["Low"].plot(figsize=(7, 5), linewidth=2.5, color='maroon')

plt.xlabel("Date",labelpad = 15)

plt.ylabel("Low",labelpad = 15)

plt.title("Moving_Low_Price", y=1.02, fontsize=22);
fb = data.copy()

fb = fb.drop(["Open","High","Low","Vol.","Change %"],axis = 1)

fb = fb.rename(columns = {"Date":"ds","Price":"y"})

fb
m = Prophet()

m.fit(fb)
future = m.make_future_dataframe(periods=7)

future.tail(7)
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
ig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)