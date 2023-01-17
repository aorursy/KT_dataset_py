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
import plotly.figure_factory as ff
# Использование cufflinks в офлайн-режиме
import cufflinks
cufflinks.go_offline()

# Настройка глобальной темы cufflinks
cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/windows-store/msft.csv')
data.head()
print(data.info())
print(data.describe())
data.isnull().sum()
data = data.dropna()
data['Date'] = pd.to_datetime(data['Date'])
print(data.head(7))
print(data.tail(7))
data['Price'] = data['Price'].str.replace('₹', '') #delete money sign
data['Price'] = data["Price"].apply(lambda x: float(x.lstrip().replace(',', '')) if x!='Free' else x) #transform string to float format
data['Price'] = data["Price"].apply(lambda x: x*0.0133681 if x!='Free' else x) #convert INR to USD
data
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
px.histogram(data, x='year', title = 'Amount of apps through the years')
fig = px.line(data.groupby('year')['No of people Rated'].sum().reset_index(), x='year', y='No of people Rated',
             title = 'Amount of people rated throught the years')
fig.show()
data1 = data.set_index('Date')
px.pie(data.Category.value_counts().reset_index(), values='Category', names='index',
      title = 'Amount of apps by Category')
category = data1.iloc[:,[3,5,6]]
categ = pd.DataFrame(category.groupby(['year', 'Category']).Category.count())
categ = categ.rename(columns={'Category':'amount'})
categ.reset_index()
fig = px.line(categ.reset_index(), x="year", y="amount", color='Category',
             title='Amount of Category apps through the years')
fig.show()
hist_labels = [data.Rating.values]
group_labels = ['Rating distribution']
fig = ff.create_distplot(hist_labels, group_labels)
fig.show()

hist_labels = [data['No of people Rated'].values]
group_labels = ['No of people rated distribution']
fig = ff.create_distplot(hist_labels, group_labels)
fig.show()

rate = pd.DataFrame(data.groupby('Rating').size().reset_index())
rate = rate.rename(columns={'Rating':'rate',0:'amount'})

px.pie(rate, values='amount', names='rate', title = 'Amount of each rate')
px.bar(data.groupby('Category').Rating.mean().reset_index(), x='Category', y='Rating', color='Rating',
       title='Mean rating by category')
data['Price2'] = np.where(data.Price=='Free', 'Free', 'Paid')
px.pie(data.Price2.value_counts().reset_index(), values = 'Price2', names='index', title='Amount of free and paid apps')
px.pie(data[data.Price=='Free'].Category.value_counts().reset_index(), values='Category', names='index',
      title = 'Amount of free apps by Category')
px.pie(data[data.Price2=='Paid'].Category.value_counts().reset_index(), values='Category', names='index',
      title = 'Amount of paid apps by Category')
px.line(data[data.Price2=='Paid'].groupby('year').size().reset_index(), x='year', y=0, title = "Amount of paid apps thtough the years")
px.bar(data[data.Price2=='Paid'].groupby('Category').Rating.mean().reset_index(), x='Category', y='Rating', 
      title = 'Mean rating for each category in paid apps')
paid = data[data.Price!='Free']
paid['Price'] = paid['Price'].astype('float')
print('Mean price of paid apps is {} $'.format(data[data.Price!='Free'].Price.mean()))
print(paid.groupby('Category').Price.mean())
px.bar(paid.groupby('Category').Price.mean().reset_index(), x='Category', y='Price', 
      title = 'Mean price for each category')
top_free = data[(data.Price=='Free')&(data['No of people Rated']>data['No of people Rated'].mean()*1.5)&(data.Rating>4)]
top_free = top_free.sort_values(['Rating', 'No of people Rated'], ascending=False).head(20)
top_free
px.pie(top_free,values='No of people Rated', hover_data=['Name', 'Category'], title='TOP 20 FREE APPS')
px.histogram(top_free, x='Category', title = 'Category distribution among top 20 free apps')
top_paid = data[(data.Price!='Free')&(data['No of people Rated']>data['No of people Rated'].mean()*0.7)&(data.Rating>4)]
top_paid = top_paid.sort_values(['Rating', 'No of people Rated'], ascending=False).head(20)
top_paid
px.pie(top_paid,values='No of people Rated', hover_data=['Name', 'Category'], title='TOP 20 PAID APPS')
px.histogram(top_paid, x='Category', title = 'Category distribution among top 20 paid apps')
top_paid['Price'] = pd.to_numeric(top_paid['Price'])
px.bar(top_paid.groupby('Category').Price.mean().reset_index(), x='Category', y='Price')
print('Mean price for TOP 20 PAID APPS is {}'.format(top_paid.Price.mean()))
