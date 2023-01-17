# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #visualisation 
import matplotlib.pyplot as plt

from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_data= pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
test_set= pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
train_data.head()
train_data.describe()
sns.pairplot(train_data)
df2= train_data.groupby(['Date','Country_Region'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()
df2.head()


countrywise= px.line(df2, x='Date', y='ConfirmedCases',color='Country_Region', title="Worldwide Confirmed/Death Cases Over Time wrt countries")
countrywise.layout.template = 'plotly_dark'
countrywise.show()
#a plot between cases as y and dates as x 
worldwide_df = train_data.groupby('Date')[['ConfirmedCases', 'Fatalities']].sum().reset_index()

worldwide_df.head()

new_df= pd.melt(worldwide_df, id_vars=['Date'], value_vars=['ConfirmedCases','Fatalities'])

worldwide = px.line(new_df, x="Date", y="value", color='variable', title="Worldwide Confirmed/Death Cases timeline")
worldwide.layout.template = 'presentation'

worldwide.show()


worldwide = px.line(new_df, x="Date", y="value", color='variable', 
              title="Worldwide Confirmed/Death Cases Over Time", log_y=True)
worldwide.layout.template = 'ggplot2'

worldwide.show()

train_china = train_data.query('Country_Region == "China"')
train_china = train_china.groupby(['Date','Province_State'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()
train_china.head()
china= px.line(train_china, x='Date', y='ConfirmedCases',color='Province_State', title="china Confirmed Cases Over Time")
china.layout.template = 'ggplot2'
china.show()
china= px.line(train_china, x='Date', y='ConfirmedCases',color='Province_State', title="china Confirmed Cases Over Time(log scale)", log_y=True)
china.layout.template = 'ggplot2'
china.show()
train_us = train_data.query('Country_Region == "US"')
train_us = train_us.groupby(['Date','Province_State'])[['ConfirmedCases', 'Fatalities']].sum().reset_index()
train_us.tail()
train_us['Date']= pd.to_datetime(train_us['Date'])
mask= (train_us['Date'] > '2020-3-8')
train_us= train_us.loc[mask]

us= px.line(train_us, x='Date', y='ConfirmedCases',color='Province_State', title="US- Confirmed Cases Over Time")
us.show()
us= px.line(train_us, x='Date', y='ConfirmedCases',color='Province_State', title="US- Confirmed Cases Over Time", log_y=True)
us.show()
train_india = train_data.query('Country_Region == "India"')
train_india= train_india.groupby('Date')[['ConfirmedCases', 'Fatalities']].sum().reset_index()
train_india.head()
df_ind= pd.melt(train_india, id_vars=['Date'], value_vars=['ConfirmedCases','Fatalities'])

india = px.line(df_ind, x="Date", y="value", color='variable', title="India- Confirmed/Death Cases timeline")
india.layout.template = 'presentation'

india.show()

##I'll be doing more analysis/ predictions after i completely study the SIR model and all other information on epidemics 