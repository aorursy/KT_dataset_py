# many of the ideas here are inspired by an EDA at https://www.kaggle.com/thykhuely

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os, re
import pickle
import collections
import random
from time import time
import math
import tensorflow as tf

import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
%matplotlib inline
df = pd.read_csv("../input/train.tsv", sep = '\t')
df.head()
# perform some cleaning of the text fields: remove non-characters, make lower cases, splitting item category into main and sub categories
def clean(text):
    return re.sub(r'[^\w\s]','',text)
def lower(text):
    return text.lower()
# general categories
def split_cat(text): # credit to https://www.kaggle.com/thykhuely
    cats = text.split("/")
    if len(cats) >=3:
        return cats[0:3]
    else: return ("No Label", "No Label", "No Label") 

for column in ['name', 'brand_name', 'item_description']:
    df[column] = df[column].astype(str) 
    df[column] = df[column].apply(clean).apply(lower)
df['category_name'] = df['category_name'].astype(str).apply(lower)
df['general_cat'], df['subcat_1'], df['subcat_2'] = zip(*df['category_name'].apply(lambda x: split_cat(x)))
df['log_price'] = np.log(df['price']+1)
df.head()
# visualize share of items by main category
# code adapted from https://www.kaggle.com/thykhuely/mercari-interactive-eda-topic-modelling
df['general_cat'], df['subcat_1'], df['subcat_2'] = zip(*df['category_name'].apply(lambda x: split_cat(x)))
x = df['general_cat'].value_counts().index.values.astype('str') 
y = df['general_cat'].value_counts().values 
labels = [cat for cat in x] 
values = [value for value in y] 
trace = go.Pie(labels = labels, values = values) 
layout = go.Layout(title="Share of Items by Main Category")
fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
# code adapted from https://www.kaggle.com/thykhuely/mercari-interactive-eda-topic-modelling
x = df['subcat_1'].value_counts().sort_values(ascending=False).index.values[:15].astype('str')
y = df['subcat_1'].value_counts().sort_values(ascending=False).values[:15]
labels = [cat for cat in x]
values = [value for value in y]
trace = go.Pie(labels = labels, values = values)
layout = go.Layout(title="Share of Items by Sub Category")
fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig, filename = "Share of Items by Sub Category")
# distribution of log prices in each general category
general_cats = [cat for cat in df['general_cat'].unique()]
x = [df.loc[df['general_cat']==cat, 'log_price'] for cat in general_cats]
# x is the list of prices corresponding to items within each general category.
hist_data = x
group_labels = general_cats
fig = ff.create_distplot(hist_data, group_labels, show_hist = False, show_rug = False)
fig.layout.xaxis.update({'domain': [1, 6]})
fig.layout.update(title='Distribution of log prices in general categories')
py.iplot(fig, filename = "Distribution of log prices in general categories")
# the distribution of average log price for each category name
cat_avg_price = df.groupby(['category_name']).log_price.agg(['mean']).reset_index()
cat_avg_price = cat_avg_price.rename(columns={"mean": "cat_avg_log_price"})
trace = go.Histogram(x=cat_avg_price['cat_avg_log_price'])
data = [trace]
layout = go.Layout(title="Distribution of average log price in each category name")
fig = go.Figure(data=data, layout=layout)
plot = py.iplot(fig)
# pushing the granularity further by looking at combinations of category name, brand name, and shipping
cat_avg_price = df.groupby(['category_name', 'brand_name', 'shipping']).log_price.agg(['mean']).reset_index()
cat_avg_price = cat_avg_price.rename(columns={"mean": "cat_avg_log_price"})
trace = go.Histogram(x=cat_avg_price['cat_avg_log_price'])
data = [trace]
layout = go.Layout(title="Distribution of average log price in each category name")
fig = go.Figure(data=data, layout=layout)
plot = py.iplot(fig)
trace1 = go.Histogram(x=cat_avg_price[cat_avg_price.shipping==0]['cat_avg_log_price'],
    histnorm='count',
    name='Without Shipping',
    xbins=dict( start=0,end=7.0, size=0.1),
    marker=dict(color='#FFD7E9',),
    opacity=0.75
)
trace2 = go.Histogram(
    x=cat_avg_price[cat_avg_price.shipping==1]['cat_avg_log_price'],
    name='With Shipping',
    xbins=dict(start=0,end=7.0,size=0.1),
    marker=dict(color='#EB89B5'),
    opacity=0.75
)
data = [trace1, trace2]
layout = go.Layout(
    title='Comparison of Average Log Prices in Each Category by Shipping Included Status',
    xaxis=dict(title='Value'),
    yaxis=dict(title='Count'),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# looks like prices with shipping are generaly lower. note that because this is a histogram, we are not comparing prices within a given category, 
# though that can be done with simple t-tests
# suggests that a predictive model of prices should take into account the category, brand name, as well as whether shiping fee is included