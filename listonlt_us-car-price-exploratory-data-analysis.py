import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins

%config InlineBackend.figure_format = 'retina' 

plt.rcParams['figure.figsize'] = 8, 5

pd.options.mode.chained_assignment = None 

pd.set_option('display.max_columns',None)

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')

df.drop('Unnamed: 0',axis = 1, inplace = True)

df.head()
df.shape
df.describe()
#missing data

df.isnull().sum().sort_values(ascending=False)
median_price = df['price'].median()

df['price'] = df['price'].astype(int)

df['price'].replace(0,median_price ,inplace=True)
cars_per_brand = df.groupby('brand')['model'].count().reset_index().sort_values('model',ascending = False).head(10)

cars_per_brand = cars_per_brand.rename(columns = {'model':'count'})

fig = px.bar(cars_per_brand, x='brand', y='count', color='count')

fig.show()
cars_by_model_year = df.groupby('year')['model'].count().reset_index().sort_values('model',ascending = False)

cars_by_model_year = cars_by_model_year[cars_by_model_year['year'] >= 2010]

cars_by_model_year = cars_by_model_year.rename(columns = {'model':'count'})

fig = px.bar(cars_by_model_year, x='year', y='count', color='count')

fig.show()
car_colors = df.groupby('color')['model'].count().reset_index().sort_values('model',ascending = False).head(10)

car_colors = car_colors.rename(columns = {'model':'count'})

fig = px.bar(car_colors, x='color', y='count', color='count')

fig.show()
cars_per_state = df.groupby('state')['model'].count().reset_index().sort_values('model',ascending = False).head(10)

cars_per_state = cars_per_state.rename(columns = {'model':'count'})

fig = px.bar(cars_per_state, x='state', y='count', color='count')

fig.show()
expensive_cars = df.sort_values('price',ascending = False).head(2)

fig = px.bar(expensive_cars, x='brand', y='price', color='price')

fig.show()
sns.distplot(df['price']);
#skewness and kurtosis

print("Skewness: %f" % df['price'].skew())

print("Kurtosis: %f" % df['price'].kurt())
data = df[['price','year']]

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x='year', y="price", data=data)
#correlation matrix

corrmat = df.corr()

f, ax = plt.subplots(figsize=(8, 8))

sns.heatmap(corrmat,annot = True);
#scatterplot

sns.set()

sns.pairplot(df, size = 2)

plt.show();