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
import scipy.stats
import pylab

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.offline as pyoff
pio.templates.default = "plotly_white"


import warnings
warnings.filterwarnings("ignore")
import pandas as pd

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.graphics.tsaplots as sgt 
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
sns.set()
train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv", parse_dates=["Date"], index_col='Date')
df_train = train.copy()
df_train.describe()
df_train.head(5)
train.ConfirmedCases.plot(figsize = (15,6), marker = "*", color = 'teal',linestyle = 'dashed', linewidth =1)
plt.title("Total confirmed case of covid19")
plt.xticks(rotation=90)
plt.show()

train.Fatalities.plot(figsize = (15,6), marker = "*", color = 'crimson',linestyle = 'dashed', linewidth =1)
plt.title("Total fatalities of covid19")
plt.xticks(rotation=90)
plt.show()
train.ConfirmedCases.plot(figsize = (10,5),marker="^", color = 'green', linestyle = 'dashdot',  title = "total ConfirmedCases")
train.Fatalities.plot(figsize = (10,5),marker="*", color = 'fuchsia',linestyle = 'dashdot',title = "total Fatalities")
plt.xlabel('Date')
plt.ylabel('confirmed and fatalities cases')
plt.title('confirmed vs fatalities')
plt.legend()
plt.xticks(rotation=90)
plt.show()
scipy.stats.probplot(train.ConfirmedCases, plot = pylab)
plt.title("QQ Plot of confirmed cases", size = 20)
pylab.show()
scipy.stats.probplot(train.Fatalities, plot = pylab)
plt.title("QQ Plot of fatalities ", size = 20)
pylab.show()
wn_confirmed= np.random.normal(loc = train.ConfirmedCases.mean(), scale = train.ConfirmedCases.std(), size = len(train))
wn_fatalities = np.random.normal(loc = train.Fatalities.mean(), scale = train.Fatalities.std(), size = len(train))
train['wn_confirmed'] = wn_confirmed
train['wn_fatalities'] = wn_fatalities
train.wn_confirmed.plot(figsize = (10,5), linestyle = 'dotted', color = 'coral')
plt.title("White Noise Time-Series of confirmed case",  size= 24)
plt.xticks(rotation=90)
plt.show()
train.wn_fatalities.plot(figsize = (10,5), linestyle = 'dotted', color = 'indigo')
plt.title("White Noise Time-Series of fatalities case",  size=24)
plt.xticks(rotation=90)
plt.show()
train.wn_confirmed.plot(figsize = (10,5), title = "total white noise", marker = "*")
train.ConfirmedCases.plot(figsize = (10,5), title = "total ConfirmedCases", marker = "*")
plt.xlabel('Date')
plt.ylabel('confirmed vs white noise')
plt.title('confirmed vs white noise')
plt.legend()
plt.xticks(rotation=90)
plt.show()
train.wn_fatalities.plot(figsize = (10,5), title = "total white noise", color= "red", linestyle = 'dashed', marker = "^")
train.Fatalities.plot(figsize = (10,5), title = "total Fatalities", color = 'gold', linestyle = 'dashed', marker = '^')
plt.xlabel('Date')
plt.ylabel('Fatalities vs white noise')
plt.title('Fatalities vs white noise')
plt.legend()
plt.xticks(rotation=90)
plt.show()
sts.adfuller(train.ConfirmedCases)
sts.adfuller(train.Fatalities)
sdec_confirmed  = seasonal_decompose(train.ConfirmedCases, model = "additive", freq = 30)
sdec_confirmed.plot()
plt.title("seasonal decom. of conformed cases")
plt.show()
sdec_fatalities = seasonal_decompose(train.Fatalities, model = "additive", freq = 30)
sdec_fatalities.plot()
plt.title("seasonal decom. of fatalities ")
plt.show()

sgt.plot_acf(train.ConfirmedCases, lags = 40, zero = False)
plt.title("ACF ConfirmedCases", size = 24)
plt.show()

sgt.plot_acf(train.Fatalities, lags = 40, zero = False)
plt.title("ACF Fatalities", size = 24)
plt.show()
sgt.plot_pacf(train.ConfirmedCases, lags = 40, zero = False, method = ('ols'))
plt.title("PACF ConfirmedCases", size = 24)
plt.show()

sgt.plot_pacf(train.Fatalities , lags = 40, zero = False, method = ('ols'))
plt.title("PACF Fatalities", size = 24)
plt.show()
train_new = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
train_new['Date_str']=train_new['Date'].copy()
train_new['Date']=pd.to_datetime(train_new['Date'])
total_df = train_new.groupby(['Date','Date_str', 'Country_Region'])['ConfirmedCases', 'Fatalities'].sum()
total_df.head()
total_df = total_df.reset_index()
total_df.head()
fig = px.scatter_geo(total_df,  locations="Country_Region",
                     locationmode='country names',
                     color="Country_Region", 
                     hover_name="ConfirmedCases", 
                     size="ConfirmedCases", 
                     title='Total ConfirmedCases over time',
                      
                     projection="natural earth")
fig.show()
fig = px.scatter_geo(total_df,  locations="Country_Region",
                     locationmode='country names',
                     color="Country_Region", 
                     hover_name="Fatalities", 
                     size="Fatalities", 
                     title='Total fatalities over time',
                      
                     projection="natural earth")
fig.show()
fig = px.line(total_df, x='Date', y='ConfirmedCases')
fig.show()

fig = px.line(total_df, x='Date', y='Fatalities')
fig.show()
fig = px.line(total_df, x='Date', y='ConfirmedCases', title='Time Series with Range Slider and Selectors')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="2m", step="month", stepmode="backward"),
            dict(step="all"),
            
        ])
    )
)
fig.show()

fig = px.line(total_df, x='Date', y='Fatalities', title='Time Series with Range Slider and Selectors')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="2m", step="month", stepmode="backward"),
            dict(step="all"),
            
        ])
    )
)
fig.show()