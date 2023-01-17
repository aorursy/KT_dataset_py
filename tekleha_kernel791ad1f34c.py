# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
import keras

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

color = sns.color_palette()
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import warnings
warnings.filterwarnings('ignore')
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm
# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/global-commodity-trade-statistics/commodity_trade_statistics_data.csv')
df.head()
df.isnull().sum()
df_ie2=df.groupby(['year'],as_index=True)['weight_kg','quantity'].agg('sum')
df_ie=df.groupby(['year'],as_index=False)['weight_kg','quantity'].agg('sum')
df_ie2.head(1)
df_ie.head(1)
df_ie2.plot(figsize=(12,6))
temp1 = df_ie[['year', 'weight_kg']] 
temp2 = df_ie[['year', 'quantity']] 
# temp1 = gun[['state', 'n_killed']].reset_index(drop=True).groupby('state').sum()
# temp2 = gun[['state', 'n_injured']].reset_index(drop=True).groupby('state').sum()
trace1 = go.Bar(
    x=temp1.year,
    y=temp1.weight_kg,
    name = 'Year with Import/Export in terms of Weight (Kg.)'
)
trace2 = go.Bar(
    x=temp2.year,
    y=temp2.quantity,
    name = 'Year with Import/Export in terms of no. items (Quantity)'
)

fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Import/Export in terms of Weight (Kg.)', 'Year with Import/Export in terms of no. items (Quantity)'))
                                                          

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
                          
fig['layout']['xaxis1'].update(title='Year')
fig['layout']['xaxis2'].update(title='Year')

fig['layout']['yaxis1'].update(title='Year with Import/Export in terms of Weight (Kg.)')
fig['layout']['yaxis2'].update(title='Year with Import/Export in terms of no. items (Quantity)')
                          
fig['layout'].update(height=500, width=1500, title='Import/Export in terms of Weight(kg.) & No. of Items')
iplot(fig, filename='simple-subplot')
df.shape
cnt_srs = df['flow'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic'
    ),
)

layout = go.Layout(
    title='Imports/Exports Ratio'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
offline.iplot(fig, filename="Ratio")

df_country=df.groupby(['country_or_area'],as_index=False)['weight_kg','quantity'].agg('sum')
df_country=df_country.sort_values(['weight_kg'],ascending=False)
fig, ax = plt.subplots()

fig.set_size_inches(13.7, 8.27)

sns.set_context("paper", font_scale=1.5)
f=sns.barplot(x=df_country["country_or_area"].head(31), y=df_country['weight_kg'].head(31), data=df_country)
f.set_xlabel("Name of Country",fontsize=15)
f.set_ylabel("Import/Export Amount",fontsize=15)
f.set_title('Top countries dominating Global Trade')
for item in f.get_xticklabels():
    item.set_rotation(90)
# df['category'].unique()
new_category = []
for el in df['category']: 
    el = el.lower()
    if el in ['10_cereals']:
        new_category.append(1)
    else:
        new_category.append(0)
print(len(new_category))


df['category'] = new_category
df.head()
df.drop(columns = ['comm_code'], inplace = True)
df.drop(columns = ['commodity'], inplace = True)
df = df.drop(df[df['category'] == 0].index)
df = df.drop(df[df['weight_kg'] == 0].index)
df = df[df.weight_kg >= 0]
df.drop(columns = ['quantity_name'], inplace = True)
df.drop(columns = ['quantity'], inplace = True)
# df = df[df.country_or_area == 'Russian Federation']
df
# max_weight = df.weight_kg.max()
# min_weight = df.weight_kg.min()
# new_weight = []
# for el in df['weight_kg']: 
#     a = float((el - min_weight) / (max_weight - min_weight))
#     new_weight.append(round(a, 2))
# df['weight_kg'] = new_weight
df
X = df
df.drop(columns = ['country_or_area'], inplace = True)
df.drop(columns = ['flow'], inplace = True)
y = df['weight_kg']
df.drop(columns = ['weight_kg'], inplace = True)
df
# commodity_dummies = pd.get_dummies(df['commodity'], prefix='comm', dummy_na=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
y
clf = MLPClassifier(random_state=1, max_iter=10)
clf.fit(X_train, y_train)
clf
result = clf.predict(X_test)
y_test[:100]
result[:5]
plt.plot(y_test, result, 'r+')
y_test
clf.score(X_test, y_test)