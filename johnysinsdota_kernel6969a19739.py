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

from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("../input/covid19-corona-virus-india-dataset/complete.csv")
confirm=df.groupby('Date').sum()['Total Confirmed cases'].reset_index()
confirm.columns = ['ds','y']
confirm
confirm['ds']=pd.to_datetime(confirm['ds'])

m = Prophet()
m.fit(confirmed)
future = m.make_future_dataframe(periods=3)
future.tail()
forecast = m.predict(future)
confirmed_forecast_plot = m.plot(forecast)
confirmed_forecast_plot =m.plot_components(forecast)
