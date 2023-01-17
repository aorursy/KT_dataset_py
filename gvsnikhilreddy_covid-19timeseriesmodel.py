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

import seaborn as sns

import matplotlib.pylab as plt

%matplotlib inline

import plotly.express as px

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode()

from fbprophet import Prophet
data = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

data_1=data

data
data.info()
data.shape
data.columns
top_cases=data.groupby('Country_Region')['ConfirmedCases'].max().sort_values(ascending=False).to_frame()

top_cases=top_cases

top_cases.style.background_gradient(cmap='Reds')
data=data.groupby(["Date"])['ConfirmedCases'].sum().to_frame()

data=data.reset_index()
fix=px.bar(data,x="Date",y='ConfirmedCases',color="ConfirmedCases")

fix.show()
fig=py.iplot([go.Scatter(

    x=data['Date'],

    y=data['ConfirmedCases'])])           
data.columns=['ds','y']

m = Prophet(interval_width=0.95)

m.fit(data)

future = m.make_future_dataframe(periods=28)

forecast = m.predict(future)

forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
d=m.plot(forecast)

d1=m.plot_components(forecast)
forecast

dat=np.array(forecast["yhat"])

len(dat)
data_1=data_1.groupby(["Date"])['Fatalities'].sum().to_frame()

data_1=data_1.reset_index()

data_1
fix=px.bar(data_1,x="Date",y='Fatalities',color='Fatalities')

fix.show()
data_1.columns=['ds','y']

m = Prophet(interval_width=0.95)

m.fit(data_1)

future = m.make_future_dataframe(periods=28)

forecast = m.predict(future)

forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
d=m.plot(forecast)
d1=m.plot_components(forecast)
dat_1=np.array(forecast["yhat"])

len(dat_1)   

ds=np.array(forecast['ds'])

len(ds)
data=pd.DataFrame({'ConfirmedCases':dat,'Fatalities':dat_1})

data.index.name='ForecastId'

data=data.reset_index()

data
data.to_csv('submission.csv',index=False )