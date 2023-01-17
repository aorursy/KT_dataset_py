# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
import plotly.graph_objs as go
from plotly.offline import iplot

# Использование cufflinks в офлайн-режиме
import cufflinks
cufflinks.go_offline()

# Настройка глобальной темы cufflinks
cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/onlineretail/OnlineRetail.csv', encoding='ISO-8859-1')
data.head()
data.isnull().sum()
data.info()
print('Quantity column')
print(data.Quantity.describe())
print('UnitPrice column')
print(data.UnitPrice.describe())
data = data[(data.Quantity>0)&(data.UnitPrice>0)]
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data = data.dropna(subset=['CustomerID'])
from datetime import datetime, date
datelist = pd.date_range(start="2010-12-01",end="2011-12-09")
for i in data['InvoiceDate']:
    i = datetime.date(i)
    if i not in datelist:
        print(i)
data['TotalPrice'] = data['Quantity']*data['UnitPrice']
data['Year'] = pd.DatetimeIndex(data['InvoiceDate']).year
data['Month'] = pd.DatetimeIndex(data['InvoiceDate']).month
data.groupby(['Year', 'Month']).InvoiceNo.count().plot(kind='bar', title='Amount of invoices per month')
data.groupby(['Year', 'Month']).CustomerID.count().plot(kind='bar', title='Amount of customers per month')
px.bar(data[['InvoiceDate','TotalPrice']].set_index('InvoiceDate').resample('M').sum().reset_index(),
       x='InvoiceDate', y='TotalPrice', title = 'Total Revenue per month')
a=data[data.Year==2011].groupby('Month').InvoiceNo.nunique().reset_index()
b=data[data.Year==2011].groupby('Month').TotalPrice.sum().reset_index()
a=a.merge(b, right_on='Month', left_on='Month', how='inner')
px.scatter(a, x='InvoiceNo', y='TotalPrice', hover_data=['Month'], title = 'Amount of invoices per month and total revenue distribution')
px.pie(data.groupby('Country').TotalPrice.sum().reset_index()[:20], values='TotalPrice', names='Country', 
      title='TOP BEST 20 COUNTRIES BY SALES')
px.pie(data.groupby('Country').TotalPrice.sum().reset_index()[20:], values='TotalPrice', names='Country', 
      title='THE WORST 20 COUNTRIES BY SALES')
px.bar(data[['InvoiceDate','TotalPrice']].set_index('InvoiceDate').resample('W').sum().reset_index(),
       x='InvoiceDate', y='TotalPrice')
data['Hour'] = data['InvoiceDate'].dt.hour
data['WeekDay']=data['InvoiceDate'].dt.weekday
data['WeekDay'] = data['WeekDay'].replace({0:'Mon', 1:'Thu',2:'Wed', 3:'Thur', 4:'Fri', 5:'Sat', 6:'Sun'})
px.bar(data.groupby('WeekDay').TotalPrice.sum().reset_index(), x='WeekDay', y='TotalPrice')
px.bar(data.groupby('Hour').TotalPrice.sum().reset_index(), x='Hour', y='TotalPrice')
