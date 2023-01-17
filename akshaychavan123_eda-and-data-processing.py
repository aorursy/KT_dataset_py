import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import plotly.express as px

import plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.tsa.arima_model import ARIMA

from progressbar import *



from sklearn.preprocessing import LabelEncoder



from math import sqrt



# Function for splitting training and test set

from sklearn.model_selection import train_test_split



# Function to perform data standardization 

from sklearn.preprocessing import StandardScaler



# Import classes for ML Models

from sklearn.linear_model import Ridge  ## Linear Regression + L2 regularization

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



# Evaluation Metrics

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error as mae





from keras.models import Sequential, Model

from keras import optimizers

from keras.layers import Dense

import tensorflow as tf



from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import LSTM, Dense, RepeatVector, Masking, TimeDistributed

from tensorflow. keras.utils import plot_model



from tensorflow.keras.optimizers import Adam



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/iowa-liquor-sales/Iowa_Liquor_Sales.csv")
df.head()
# Last 5 rows

df.tail()
df.shape
df.columns
# Checknig the types of columns

df.dtypes
df.info()
# Displaying unique names of country

print(df['City'].nunique())

df['City'].unique()
df['City'].value_counts()
# Checking for the Null values

df.isnull().sum()
# Dropping null values rows and again checking shape of dataframe.

df.dropna(inplace = True)

print(df.shape)
# Check for the duplicates values.

df.drop_duplicates()

df.shape
df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year

df['Month'] = df['Date'].dt.month

df['Day'] = df['Date'].dt.day
df.head()
# Printing minimum and the maximum date from dataset.

print(df['Date'].min())

print(df['Date'].max())
print(df['Category Name'].nunique())

df['Category Name'].unique()
# Replacing the dollar symbol in the sales column by None



df['Sale (Dollars)'] = df['Sale (Dollars)'].str.replace('$', '')

df['Sale (Dollars)'] = df['Sale (Dollars)'].astype('float')
df_plot = df.fillna('NA').groupby(['Category Name','Pack','Date'])['Sale (Dollars)'].sum().groupby(

            ['Category Name','Pack']).max().sort_values().groupby(

            ['Category Name']).sum().sort_values(ascending=False)

top_count = pd.DataFrame(df_plot)

top_count1 = pd.DataFrame(df_plot.head(10))
import plotly.graph_objects as go



fig_reg = px.bar(top_count1,x=top_count1.index, y='Sale (Dollars)',color='Sale (Dollars)')

fig_reg.update_layout(

    title="Sales of liquor per category",

    xaxis_title=" Category Name",

    yaxis_title="Sales in dollars",

    )

fig_reg.show()
df_plot = df.fillna('NA').groupby(['City','Pack','Date'])['Sale (Dollars)'].sum().groupby(

            ['City','Pack']).max().sort_values().groupby(

            ['City']).sum().sort_values(ascending=False)

top_count1 = pd.DataFrame(df_plot)

top_count1 = pd.DataFrame(df_plot.head(20))



fig_reg = px.bar(top_count1,x=top_count1.index, y='Sale (Dollars)',color='Sale (Dollars)')

fig_reg.update_layout(

    title="Sales of liquor per city",

    xaxis_title=" City Name",

    yaxis_title="Sales in dollars",

    )

fig_reg.show()
df_plot = df.fillna('NA').groupby(['Month','Pack','Date'])['Sale (Dollars)'].sum().groupby(

            ['Month','Pack']).max().sort_values().groupby(

            ['Month']).sum().sort_values(ascending=False)

top_count1 = pd.DataFrame(df_plot)

top_count1 = pd.DataFrame(df_plot.head(50))



fig_reg = px.bar(top_count1,x=top_count1.index, y='Sale (Dollars)',color='Sale (Dollars)')

fig_reg.update_layout(

    title="Sales of liquor per Month",

    xaxis_title=" Month Number",

    yaxis_title="Sales in dollars",

    )

fig_reg.show()
daily_sales = df.groupby('Date', as_index=False)['Sale (Dollars)'].sum()
daily_sales_sc = go.Scatter(x=daily_sales['Date'], y=daily_sales['Sale (Dollars)'])

layout = go.Layout(title='Daily sales', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))

fig = go.Figure(data=[daily_sales_sc], layout=layout)

iplot(fig)
df_plot = df.fillna('NA').groupby(['Category Name','Pack','Date'])['Sale (Dollars)'].sum().groupby(

            ['Category Name','Pack']).max().sort_values().groupby(

            ['Category Name']).sum().sort_values(ascending=False)

top_count1 = pd.DataFrame(df_plot)

#top_count1 = pd.DataFrame(df_plot.head(10))



df_plot = df.fillna('NA').groupby(['Category Name','Pack','Date'])['Volume Sold (Liters)'].sum().groupby(

            ['Category Name','Pack']).max().sort_values().groupby(

            ['Category Name']).sum().sort_values(ascending=False)

top_count2 = pd.DataFrame(df_plot)

#top_count2 = pd.DataFrame(df_plot.head(10))
# Ordrening the countries by number of fatalities

top_count = pd.concat([top_count1 , top_count2],axis=1)

top_count = top_count.sort_values(['Sale (Dollars)'],ascending=False)[:10]

top_count
fig = go.Figure(data=[

    go.Bar(name='sale in dollars',x=top_count.index, y=top_count['Sale (Dollars)']),

    go.Bar(name='Volume in litres',x=top_count.index, y=top_count['Volume Sold (Liters)'])

])

# Change the bar mode

fig.update_layout(barmode='group',title="Sales of liquors with category and the volume of liquor sold.",

    xaxis_title=" category",

    yaxis_title="Sale and the amount of liquor sold in litre.",)

fig.show()

df_plot = df.fillna('NA').groupby(['Vendor Name','Pack','Date'])['Volume Sold (Liters)'].sum().groupby(

            ['Vendor Name','Pack']).max().sort_values().groupby(

            ['Vendor Name']).sum().sort_values(ascending=False)

top_count1 = pd.DataFrame(df_plot)

#top_count1 = pd.DataFrame(df_plot.head(50))



fig_reg = px.bar(top_count1,x=top_count1.index, y='Volume Sold (Liters)',color='Volume Sold (Liters)')

fig_reg.update_layout(

    title="Volume sold by vendor name",

    xaxis_title=" Vendor Name",

    yaxis_title="Liquor sold in Litres",

    )

fig_reg.show()