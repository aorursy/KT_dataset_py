# Importing libraries

import os

import warnings

warnings.filterwarnings('ignore')

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight') 

# Above is a special style template for matplotlib, highly useful for visualizing time series data

%matplotlib inline

from pylab import rcParams

from plotly import tools

import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import statsmodels.api as sm

from numpy.random import normal, seed

from scipy.stats import norm

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_process import ArmaProcess

from statsmodels.tsa.arima_model import ARIMA

import math

from sklearn.metrics import mean_squared_error

print(os.listdir("../input"))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/property-sales/raw_sales.csv")
data.head()
data.info()
data.isnull().any().any()
print("Time period from {} to {}".format(data.datesold.min(), data.datesold.max()))
# Monthly number of house sales between 2007 and 2019

pd.to_datetime(data.datesold).dt.month.value_counts().plot('bar')
# Yearly number of house sales between 2007 and 2019

pd.to_datetime(data.datesold).dt.year.value_counts().plot('bar')
# We bin the postcode for easier analysis

bins = pd.IntervalIndex.from_tuples([(2600, 2700), (2701, 2800), (2801, 2915)])

data['postcode_bin'] = pd.cut(data['postcode'],bins)
sns.countplot(data['postcode_bin'])
data.propertyType.value_counts()
plt.pie(data['propertyType'].value_counts(), labels=['house','unit'], autopct='%1.1f%%', startangle = 150)

plt.axis('equal')
data['datesold']= pd.to_datetime(data['datesold'])
from pandas import Interval



# House sales in postcode 2600 - 2700

data1 = data[data.postcode_bin == Interval(2600, 2700, closed='right')]



# House sales in postcode 2801-2915

data2 = data[data.postcode_bin == Interval(2801, 2915, closed='right')]
# Average sale price of houses for each of the two postcode bins

rcParams['figure.figsize'] = 20,5 

data1.groupby('datesold').price.mean().plot()

data2.groupby('datesold').price.mean().plot()

plt.legend(['2600-2700 postcode', '2801-2915 postcode'])

plt.xlabel("Year")

plt.ylabel("Average Price")
sm.tsa.seasonal_decompose(data1.groupby('datesold').price.mean(), freq=365).plot()
sm.tsa.seasonal_decompose(data2.groupby('datesold').price.mean(), freq=365).plot()
# Auotcorrelation for house sales price for postcodes group 1

plot_acf(data1["price"], lags=25, title="postcodes 2600-2700")
# Auotcorrelation of house sale price for postcodes group 2



plot_acf(data2["price"], lags=25, title="postcodes 2801-2915")
data_house = data[data.propertyType == 'house'] 

data_unit = data[data.propertyType == 'unit']
# Average Sale price of house property type over years

data_house.groupby(['datesold']).price.mean().plot()
sm.tsa.seasonal_decompose(data_house.groupby('datesold').price.mean(), freq=365).plot()
# Average Sale price of units property type over years

data_unit.groupby(['datesold']).price.mean().plot()
sm.tsa.seasonal_decompose(data_unit.groupby('datesold').price.mean(), freq=365).plot()
data['datesold_year'] = data['datesold'].dt.year

sns.boxplot(x= 'datesold_year', y = 'price', data=data)