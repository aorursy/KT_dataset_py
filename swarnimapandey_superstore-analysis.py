##import necessary packages 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import warnings

import itertools

warnings.filterwarnings(action='ignore')

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima_model import ARIMA

from pylab import rcParams

import statsmodels.api as sm
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
#uploading the data

sales_data=pd.read_excel('../input/superstore/Sample - Superstore.xlsx')

sales_data.head()
sales_data.tail() ## viewing last 4 records
sales_data.shape # viewing the shape of the data
sales_data.info() # viewing the types of each feature in the dataset
sales_data.describe() # getting some basic overview
sales_data.columns #All Columns 
sales_data.isnull().sum()
sales_data=sales_data.dropna(axis=0) # since only 11 records (that too of continuous variable) were missing so it is efficient to use dropna here.
sales_data.isnull().sum()
sales_data.shape #new shape
corrmat=sales_data.corr()

top_corr=corrmat.index

plt.figure(figsize=(15,15))

#plot the heatmap

g=sns.heatmap(sales_data[top_corr].corr(),annot=True,cmap='RdYlGn')
sales_data.drop(['Row ID','Ship Date','Ship Mode','Customer ID','Postal Code','Order ID','Profit','Discount'],axis=1,inplace=True) 
## Downloading this pre-processed data into train1.csv

sales_data.to_csv('train1.csv')
print(sales_data.shape)

sales_data.head() #new shape
## Determining the categories in the Country column of the dataframe

sales_data['Country'].unique()
## Determining the number of States in US

states=sales_data['State'].unique()

np.count_nonzero(states)
## Determining the number of Cities in US

cities=sales_data['City'].unique()

np.count_nonzero(cities)
## the top 20 cities with high sales

top_cities= sales_data['City'].value_counts().nlargest(20)

top_cities
#Most frequent customers

top_customers= sales_data['Customer Name'].value_counts().nlargest(20)

top_customers
rslt_df = sales_data[sales_data['Customer Name'] == 'William Brown'] 

rslt_df.head()
# determining the unique values of category column.

category=sales_data['Category'].unique()

print(category)

print(np.count_nonzero(category))
plt.rcParams['figure.figsize'] = (10, 8)

sns.barplot(x = sales_data['Category'], y = sales_data['Sales'], palette ='dark')

plt.title('The Distribution of Sales in each Category', fontsize = 10)

plt.xlabel('Category', fontsize = 15)

plt.ylabel('Count', fontsize = 15)
# determining the total count of sub-categories/ products in the Supermarket Store

subcategory=sales_data['Sub-Category'].unique()

print(subcategory)

print(np.count_nonzero(subcategory))

#There are 17 products/ sub-categories.
# visualizing sub-category wise distribution of sales

plt.rcParams['figure.figsize'] = (19, 8)

sns.barplot(x = sales_data['Sub-Category'], y = sales_data['Sales'], palette ='dark')

plt.title('The Distribution of Sales in each Sub-Category', fontsize = 30)

plt.xlabel('Sub-Category', fontsize = 15)

plt.ylabel('Count', fontsize = 15)
#top 5 products highly in demand

top_products= sales_data['Sub-Category'].value_counts().nlargest(5)

top_products
# determining segments of customers

segment=sales_data['Segment'].unique()

print(segment)

print(np.count_nonzero(segment))
# visualizing Segment wise distribution of sales

plt.rcParams['figure.figsize'] = (19, 8)

sns.barplot(x = sales_data['Segment'], y = sales_data['Sales'], palette ='dark')

plt.title('The Distribution of Sales in each Segment', fontsize = 30)

plt.xlabel('Segment', fontsize = 15)

plt.ylabel('Count', fontsize = 15)

#visualizing state-wise sales distribution

sales_data.groupby(['State'])['Sales'].nunique().plot.bar(figsize = (19, 7), cmap= 'rainbow')

plt.gcf().autofmt_xdate()

plt.title('Comparing statewise sales frequency', fontsize = 30)

plt.xlabel('States in US', fontsize = 10)

plt.ylabel('Sales Frequency')

plt.show()

#top 10 states with high sales

top_states= sales_data['State'].value_counts().nlargest(10)

top_states
print(sales_data['State'].max()) # california is with high frequency sales whereas Wyoming has the overall maximum sale price.
plt.rcParams['figure.figsize'] = (15, 8)

sns.distplot(sales_data['Sales'], color = 'red')

plt.title('The Distribution of Sales', fontsize = 30)

plt.xlabel('Range of Sales', fontsize = 15)

plt.ylabel('No. of Sales count', fontsize = 15)

plt.show()
sales_data['Order Date'] = pd.to_datetime(sales_data['Order Date'], errors = 'coerce') # it was already datetime object before, not a necessary step
#extracting Year out of the Date to do year-wise analysis

sales_data['Year'] = sales_data['Order Date'].dt.year
#extracting month out of the Date to do month-wise analysis

sales_data['Month'] = sales_data['Order Date'].dt.month
#extracting Day out of the Date to do daywise analysis

sales_data['Date'] = sales_data['Order Date'].dt.day
sales_data.columns
# separating dependent and independent featurea

X=sales_data.copy()

X.drop(['Sales'],axis=1,inplace=True)

X.head() # independent features
y=sales_data.iloc[:,11] # target as well as dependent feature

y.head()
## visualizing through boxplot

plt.rcParams['figure.figsize'] = (19, 8)

sns.boxplot(x = sales_data['Year'], y = sales_data['Sales'], palette ='dark')

plt.title('The Distribution of Sales in each Year', fontsize = 30)

plt.xlabel('Year', fontsize = 15)

plt.ylabel('Sales Price', fontsize = 15)

year_max=sales_data[sales_data['Sales'] == 22638.480000] 

year_max
# visualizing month-wise sales distribution

plt.rcParams['figure.figsize'] = (19, 8)

sns.barplot(x = sales_data['Month'], y = sales_data['Sales'], palette ='pastel')

plt.title('The Distribution of Sales in each month', fontsize = 30)

plt.xlabel('Months', fontsize = 15)

plt.ylabel('Sales', fontsize = 15)
#visualizing daywise sales distribution

plt.rcParams['figure.figsize'] = (19, 8)

sns.barplot(x = sales_data['Date'], y = sales_data['Sales'], palette ='colorblind')

plt.title('The Distribution of Sales in each day', fontsize = 30)

plt.xlabel('Days', fontsize = 15)

plt.ylabel('Sales', fontsize = 15)
# extracting them in separate dataframe

features=['Order Date','Sales']

salesplot=sales_data[features]

salesplot.head()

salesplot.sort_values(by='Order Date',inplace=True)

salesplot
Order_date=salesplot['Order Date']

Sales=salesplot['Sales']
##Simple Scatter Plot

plt.plot_date(Order_date,Sales,xdate=True)

plt.gcf().autofmt_xdate()

plt.title('Sales Data')

plt.xlabel('Order Date')

plt.ylabel('Sales')
##interactive visualization using plotly

import plotly.express as px



fig = px.line(salesplot, x=Order_date, y=Sales, title='Time Series with Rangeslider')



fig.update_xaxes(rangeslider_visible=True)

fig.show()
# loading the pre-processed data that we prepared in the EDA notebook- train1.csv

df1=pd.read_csv('train1.csv')

df1.head()
df1.shape
## for sales forecasting we only need Order Date and Sales coulmn of the train1.csv

features=['Order Date','Sales']

dfs=df1[features]

dfs.head()
dfs.shape
dfs.info()
dfs.tail()
#converting into datetime type

dfs['Order Date'] = pd.to_datetime(dfs['Order Date'], errors = 'coerce')

dfs.info()
#setting index

dfs=dfs.groupby('Order Date')['Sales'].sum().reset_index()
dfs
dfs=dfs.set_index('Order Date')

dfs.index
#using start of each month as timestamp

y=dfs['Sales'].resample('MS').mean()
y['2015':]
#visualising Sales Time Series Data

y.plot(figsize=(15,6))

plt.show()
rcParams['figure.figsize']=19,9
decomp=sm.tsa.seasonal_decompose(y,model='additive')

fig=decomp.plot()

plt.show()
p=d=q=range(0,2)

pdq=list(itertools.product(p,d,q))

seas_pdq=[(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]
print('Some of the parameter combinations for Seasonal ARIMA:-')

print('SARIMAX: {} x {}'.format(pdq[1], seas_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seas_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seas_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seas_pdq[4]))
#using grid search to find the optimal set of parameters that yields the best performance for our model

#parameter selection for our model

for param in pdq:

    for param_seasonal in seas_pdq:

        try:

            mod=sm.tsa.statespace.SARIMAX(y,order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)

            results=mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))

        except:

            continue
mod=sm.tsa.statespace.SARIMAX(y,order=(0,1,1),seasonal_order=(0,1,1,12), enforce_stationarity=False, enforce_invertibility=False)

results=mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))

plt.show()
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = y['2014':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')

ax.set_ylabel('Product Sales')

plt.legend()

plt.show()

y_forecasted = pred.predicted_mean

y_truth = y['2017-01-01':]
# mse

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
# rmse

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
## This shows forecasts for next few years

pred_uc1 = results.get_forecast(steps=100)

pred_ci1 = pred_uc1.conf_int()

ax = y.plot(label='observed', figsize=(14, 7))

pred_uc1.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci1.index,

                pred_ci1.iloc[:, 0],

                pred_ci1.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Furniture Sales')

plt.legend()

plt.show()
## This shows the forecasts for next 7 days

pred_uc = results.get_forecast(steps=7)

pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Product Sales')

plt.title('Weekly sales forecast', fontsize=12)

plt.legend()

plt.show()
predicted_sale= pred_ci1[:7]
predicted_sale
predicted_sale.to_csv('weeklyoutput_forecast.csv')
y_truth.head()
y_forecasted.head()
output = pd.concat([y_truth, y_forecasted], axis=1)
output.head()
s1=pd.Series(output['Sales'],name='Confirmed Sales')

s2=pd.Series(output[0],name='Forecasted Sales')

df_output = pd.concat([s1, s2], axis=1)

df_output
df_output.to_csv('forecast_comparison_output2.csv')