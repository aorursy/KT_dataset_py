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

import numpy as np

import seaborn as sns

import sklearn as sk

from sklearn import metrics
path = r'/kaggle/input/daily-power-generation-in-india-20172020/file.csv'

path_to = r'/kaggle/input/daily-power-generation-in-india-20172020/State_Region_corrected.csv'



data_file = pd.read_csv(path)

data_state = pd.read_csv(path_to)
data_file.head()
data_state.head()
data_file.describe()
data_state.style.background_gradient(cmap='Reds')
from fbprophet import Prophet
energy_future = data_file.groupby('Date').sum()['Hydro Generation Estimated (in MU)'].reset_index()
energy_future.head()
energy_future.columns = ['ds','y']

energy_future['ds'] = pd.to_datetime(energy_future['ds'])
model = Prophet(interval_width=0.95)           

model.fit(energy_future)                               

future = model.make_future_dataframe(periods=365)     



energy_forecast = model.pforecast = model.predict(future)

energy_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
import plotly.offline as ply

import plotly.graph_objs as go

ply.init_notebook_mode(connected=True)
fig = go.Figure()



fig.add_trace(go.Scatter(x=energy_forecast['ds'], y=energy_forecast['yhat'], mode='lines+markers', name='Energy Required in Indian in Future',line=dict(color='Green', width=2)))

ply.iplot(fig)
fig = go.Figure()



fig.add_trace(go.Scatter(x=energy_forecast['ds'], y=energy_forecast['yhat_upper'], mode='lines+markers', name='Energy Required in Indian in Future',line=dict(color='Red', width=2)))

ply.iplot(fig)
fig = go.Figure()



fig.add_trace(go.Scatter(x=energy_forecast['ds'], y=energy_forecast['yhat_lower'], mode='lines+markers', name='Energy Required in Indian in Future',line=dict(color='Blue', width=2)))

ply.iplot(fig)