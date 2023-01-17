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
!pip install fbprophet
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import random

import seaborn as sns

from fbprophet import Prophet

%matplotlib inline
df = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")
df.head()
#Let's check for missing data



df.isnull().sum()
#drop the unnamed column since it does not contribute to our analysis



df = df.drop('Unnamed: 0',axis=1)
#chekcing the data types



df.dtypes
#convert the data column from object datatype to datetype



df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date")
#checking the initial and last dates



df['Date'].head()
plt.figure(figsize=(20,6))



plt.plot(df['Date'], df['AveragePrice']);
import pylab as pl

from pylab import rcParams

rcParams['figure.figsize'] = 12, 8
#plot Distribution of the average price



pl.figure(figsize=(15,5))

pl.title("Price Distribution")

ax = sns.distplot(df["AveragePrice"], color = 'b')
#plot violin plot of the average price vs. avocado type

sns.violinplot(y = 'AveragePrice', x = 'type', data = df);
# Bar Chart to indicate the year

plt.figure(figsize=[8,5])

sns.countplot(x = 'year', data = df)

plt.xticks(rotation = 45);
avocado_prophet_df = df[['Date', 'AveragePrice']]
avocado_prophet_df.shape
avocado_prophet_df.head()
avocado_prophet_df = avocado_prophet_df.rename(columns={'Date':'ds', 'AveragePrice':'y'})
avocado_prophet_df.shape
avocado_prophet_df.head()
m = Prophet()

m.fit(avocado_prophet_df)
# Forcasting into the future

future_complete = m.make_future_dataframe(periods=365)

forecast_complete = m.predict(future_complete)
forecast_complete[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
m.plot(forecast_complete, xlabel='Date', ylabel='Price')
m.plot_components(forecast_complete)