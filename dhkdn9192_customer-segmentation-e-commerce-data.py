# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime, nltk, warnings

import itertools

import plotly.graph_objs as go

from plotly.offline import iplot



from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

%matplotlib inline
# read the datafile

df_initial = pd.read_csv(

    '/kaggle/input/ecommerce-data/data.csv',

    encoding="ISO-8859-1", 

    dtype={'CustomerID': str,'InvoiceID': str}

)



print(f'df_initial.shape : {df_initial.shape}')
df_initial.head()
df_initial.info()
# parse datetime column

df_initial['InvoiceDate'] = pd.to_datetime(df_initial['InvoiceDate'])

df_initial.info()
# check missing values

tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column type'})

tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0:'null values (nb)'}))

tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()/df_initial.shape[0]*100).T.

                         rename(index={0:'null values (%)'}))

tab_info
df_initial.head()
# drop missing value rows

print(f'(before) df_initial.shape : {df_initial.shape}')



df_initial.dropna(axis = 0, subset = ['CustomerID'], inplace = True)

print(f'(after) df_initial.shape : {df_initial.shape}')
# check missing values

tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column type'})

tab_info = tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0:'null values (nb)'}))

tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()/df_initial.shape[0]*100).T.

                         rename(index={0:'null values (%)'}))

tab_info
# drop duplicate rows

print(f'(before) df_initial.shape : {df_initial.shape}')



df_initial.drop_duplicates(inplace = True)

print(f'(after) df_initial.shape : {df_initial.shape}')
tmp = df_initial[['CustomerID', 'InvoiceNo', 'Country']].drop_duplicates()

print(f'tmp.shape : {tmp.shape}')



countries = tmp['Country'].value_counts()

print(f'countries.shape : {countries.shape}')
data = dict(type='choropleth',

locations = countries.index,

locationmode = 'country names', z = countries,

text = countries.index, colorbar = {'title':'Order nb.'},

colorscale=[[0, 'rgb(224,255,255)'],

            [0.01, 'rgb(166,206,227)'], 

            [0.02, 'rgb(31,120,180)'],

            [0.03, 'rgb(178,223,138)'], 

            [0.05, 'rgb(51,160,44)'],

            [0.10, 'rgb(251,154,153)'], 

            [0.20, 'rgb(255,255,0)'],

            [1, 'rgb(227,26,28)']],    

reversescale = False)

#_______________________

layout = dict(title='Number of orders per country',

geo = dict(showframe = True, projection={'type':'mercator'}))

#______________

choromap = go.Figure(data = [data], layout = layout)

iplot(choromap, validate=False)
print('products     : ', df_initial['StockCode'].nunique())

print('transactions : ', df_initial['InvoiceNo'].nunique())

print('customers    : ', df_initial['CustomerID'].nunique())