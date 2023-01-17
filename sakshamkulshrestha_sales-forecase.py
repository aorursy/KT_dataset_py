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
import numpy as np

import pandas as pd

import os

import xgboost as xgb

from sklearn.model_selection import train_test_split
df= pd.read_csv('../input/demand-forecasting-kernels-only/train.csv')

test_df= pd.read_csv('../input/demand-forecasting-kernels-only/test.csv')
df.head()
df.dtypes
df.describe()


import seaborn as sns 

sns.pairplot(df)
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.title("Distribution of sales - for each item, date and store")           

ax=sns.distplot(df['sales'])
import scipy.stats as st

print("p-value for sales distribution: {}".format(st.normaltest(df.sales.values)[1]))

plt.figure(figsize=(12,5))

plt.title("Distribution of sales vs fitting normal distribution")

ax = sns.distplot(df.sales, fit= st.norm, kde=True, color='r')
store_total = df.groupby(['store'])['sales'].sum().to_frame().reset_index()

store_total.sort_values(by = ['sales'], ascending=True, inplace=True)

labels = ['Store {}'.format(i) for i in store_total.store]
#plotting the store total with the labels generated 

plt.figure(figsize=(12,5))

plt.title("sales of items per store")

ax = sns.barplot(x='store', y='sales',data=store_total, palette='Blues_d')
#total sales by item 



item_total = df.groupby(['item'])['sales'].sum().to_frame().reset_index()

item_total.sort_values(by = ['sales'], ascending=False, inplace=True)

labels = ['Item {}'.format(i) for i in item_total.item]

item_total

plt.figure(figsize=(12,7))

plt.title("total sales per item")



axis = sns.barplot(x='item', y='sales',data=item_total, palette='cubehelix')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode()

df['date']=pd.to_datetime(df['date'])

df.head()
monthly_df = df.groupby([df.date.dt.year, df.date.dt.month])['sales'].mean()

monthly_df.index = monthly_df.index.set_names(['year', 'month'])

monthly_df = monthly_df.reset_index()

x_axis = []

for y in range(13, 18):

    for m in range(1,12):

        x_axis.append("{}/{}".format(m,y))

trace = go.Scatter(x= x_axis, y= monthly_df.sales, mode= 'lines+markers', name= 'sales avg per month', line=dict(width=3))

layout = go.Layout(autosize=True, title= 'Sales - average per month', showlegend=True)

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
monthly_df = df.groupby([df.date.dt.year, df.date.dt.month])['sales'].mean()

monthly_df.index = monthly_df.index.set_names(['year', 'month'])

monthly_df = monthly_df.reset_index()

plt.figure(figsize=(15,6))

plt.title("total sales per item")



axis = sns.barplot(x='month', y='sales',data=monthly_df, palette='Blues_d')
df['train_or_test'], test_df['train_or_test'] = 'train', 'test'

data_df = pd.concat([df, test_df])

data_df.head()
# converting datetime to datetime 

data_df['date']=pd.to_datetime(data_df['date'])

data_df.dtypes

data_df.info()
data_df['year'] = data_df['date'].dt.year

data_df['quarterly'] = data_df['date'].dt.quarter

data_df['monthly'] = data_df['date'].dt.month

data_df['weekofyear'] = data_df['date'].dt.weekofyear

data_df['weekday'] = data_df['date'].dt.weekday

data_df['dayofweek'] = data_df['date'].dt.dayofweek

data_df.head()


data_df['item_quarter_mean'] = data_df.groupby(['quarterly', 'item'])['sales'].transform('mean')

data_df.head()
#columns for mean based on quarters

data_df['store_quarter_mean'] = data_df.groupby(['quarterly', 'store'])['sales'].transform('mean')

data_df['store_item_quarter_mean'] = data_df.groupby(['quarterly', 'item', 'store'])['sales'].transform('mean')

data_df.head()
#more of the same means but based on months



data_df['item_month_mean'] = data_df.groupby(['monthly', 'item'])['sales'].transform('mean')

data_df['store_month_mean'] = data_df.groupby(['monthly', 'store'])['sales'].transform('mean')

data_df['store_item_month_mean'] = data_df.groupby(['monthly', 'item', 'store'])['sales'].transform('mean')

data_df.head()
# based on weekdays



data_df['itemweekday_mean'] = data_df.groupby(['weekday', 'item'])['sales'].transform('mean')

data_df['storeweekday_mean'] = data_df.groupby(['weekday', 'store'])['sales'].transform('mean')

data_df['storeitemweekday_mean'] = data_df.groupby(['weekday', 'item', 'store'])['sales'].transform('mean')

data_df.head()
data_df.describe()
data_df.drop(['date','id','sales'],axis=1,inplace=True)
x_df= data_df[data_df['train_or_test'] == 'train']

test_df = data_df[data_df['train_or_test'] == 'train']
x_df.head()
test_df.head()
x_df.drop(['train_or_test'],axis=1,inplace=True)

test_df.drop(['train_or_test'],axis=1,inplace=True)
x_df.head()
y=pd.read_csv('../input/demand-forecasting-kernels-only/train.csv',usecols=['sales'])

y=y['sales']

pd.DataFrame(y)
from sklearn import ensemble



rfr=ensemble.RandomForestRegressor(max_depth=13, random_state=0)

rfr
%%time

rfr.fit(x_df,y)
%%time

predict=pd.DataFrame(rfr.predict(test_df),columns=['sales'])
plt.scatter(y, predict)
from sklearn import metrics 



print(metrics.r2_score(y, predict ))

print(metrics.mean_squared_error(y, predict ))

print(metrics.explained_variance_score(y, predict))
predict

ids=pd.read_csv("../input/demand-forecasting-kernels-only/test.csv",usecols=['id'])

sub=ids.join(predict)

sub.head()
sub.to_csv('result_file.csv',index=False)