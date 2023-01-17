import gc

import time

import numpy as np

import pandas as pd



from scipy.sparse import csr_matrix, hstack



from sklearn.linear_model import Ridge, Lasso

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

import lightgbm as lgb

from xgboost import XGBRegressor

import xgboost

from time import time

from sklearn.metrics import make_scorer



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

from scipy.stats import norm

import random

import time

from subprocess import check_output

print (check_output(['ls', '../input']).decode('utf8'))
train = pd.read_csv("../input/train.csv")
train_sample = train.sample(frac = 0.05)

train_sample.head()
holiday_events = pd.read_csv('../input/holidays_events.csv')

items = pd.read_csv('../input/items.csv')

oil = pd.read_csv('../input/oil.csv')

stores = pd.read_csv('../input/stores.csv')

test = pd.read_csv('../input/test.csv')



transactions = pd.read_csv('../input/transactions.csv')
print (holiday_events.head())

print (holiday_events.shape)

print (holiday_events.describe())
print (items.head())

print (items.shape)

print (items.describe())
print (len(items['family'].value_counts().index))

print (len(items['class'].value_counts().index))

print (items['perishable'].value_counts())
print (oil.head())

print (oil.shape)

print (oil.describe())
import datetime

train_sample['date'] = train_sample['date'].apply(datetime.datetime.strptime, args = ("%Y-%m-%d",))

train_sample.sort_values(by = 'date', inplace = True)

print (train_sample.head())

print (train_sample.shape)

print (train_sample.describe())

print (train_sample.isnull().sum())

train_sample['onpromotion'].fillna(value = 'missing', inplace = True)

#train_sample['onpromotion'].isnull().sum()

print (train_sample['onpromotion'].value_counts())
print (test.head())

print (test.shape)

print (test.describe())

print (test.isnull().sum())
print (test['onpromotion'].value_counts())
daily_sale = pd.DataFrame()

daily_sale['count'] = train_sample['date'].value_counts()

daily_sale['date'] = train_sample['date'].value_counts().index

daily_sale = daily_sale.sort_values(by = 'date')

print (daily_sale.head(3))

unit_sale = pd.DataFrame()

unit_sale['count'] = train_sample[train['unit_sales']>0]['unit_sales'].value_counts()

unit_sale['positive_unit_sales'] = train_sample[train['unit_sales']>0]['unit_sales'].value_counts().index

unit_sale = unit_sale.sort_values(by = 'positive_unit_sales')
promotion_count = pd.DataFrame()

promotion_count ['count'] = train_sample['onpromotion'].value_counts()
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize = (18,5))

sns.set_style('darkgrid')

ax1.plot(daily_sale['date'], daily_sale['count'])

ax1.set_xlabel('date', fontsize=15)

ax1.set_ylabel('count', fontsize=15)

ax1.tick_params(labelsize=15)

sns.barplot(x = promotion_count.index, y = promotion_count['count'],ax = ax2)

ax2.set_ylabel('count', fontsize = 15)

ax2.set_xlabel ('onpromotion',fontsize =15)

ax2.tick_params(labelsize = 15)

ax2.ticklabel_format(style = 'sci',scilimits = (0,0), axis = 'y')

plt.subplot(1,3,3)

plt.loglog(unit_sale['positive_unit_sales'], unit_sale['count'])

plt.ylabel('count', fontsize = 15)

plt.xlabel('positive_unit_sales', fontsize = 15)

plt.xticks(fontsize = 15)

plt.yticks(fontsize = 15)

plt.show()
store_count = pd.DataFrame()

store_count['count'] = train_sample['store_nbr'].value_counts().sort_index()
fig, ax = plt.subplots(figsize = (18, 3))

sns.barplot(x = store_count.index, y = store_count['count'], ax = ax)

ax.set_ylabel('count', fontsize = 15)

ax.set_xlabel('store_nbr',fontsize = 15)

ax.tick_params(labelsize=15)
item_count = pd.DataFrame()

item_count['count'] = train_sample['item_nbr'].value_counts().sort_index()

plt.plot(item_count.index)
fig, ax = plt.subplots(figsize = (18, 3))

sns.barplot(x = item_count.index, y = item_count['count'], ax = ax)

ax.set_ylabel('count', fontsize = 15)

ax.set_xlabel('item_nbr',fontsize = 15)

ax.tick_params(axis = 'x',which = 'both',top = 'off', bottom = 'off', labelbottom = 'off')
neg_unit_sale = pd.DataFrame()

neg_unit_sale['unit_sales'] = (-train_sample[train_sample['unit_sales'] < 0]['unit_sales'])

neg_unit_sale.head()
fig, ax = plt.subplots(figsize = (18, 5))

#ax.set_xscale('log')

np.log(neg_unit_sale['unit_sales']).plot.hist(ax = ax, log = True,edgecolor = 'white', bins = 50)

ax.set_xlabel('neg_unit_sales (log10 scale)', fontsize=15)

ax.set_ylabel('count', fontsize=15)

ax.tick_params(labelsize=15)
city_count = pd.DataFrame()

city_count['count'] = stores['city'].value_counts().sort_index()

fig, (ax1,ax2) = plt.subplots(1,2,figsize = (18, 4))

g = sns.barplot(x = city_count.index, y = city_count['count'], ax = ax1)

ax1.set_ylabel('count', fontsize = 15)

ax1.set_xlabel('city',fontsize = 15)

ax1.tick_params(labelsize=15)

g.set_xticklabels(city_count.index, rotation = 45)

state_count =pd.DataFrame()

state_count['count'] = stores['state'].value_counts().sort_index()

g2 = sns.barplot(x = state_count.index, y = state_count['count'], ax = ax2)

ax2.set_ylabel('count', fontsize = 15)

ax2.set_xlabel('state',fontsize = 15)

ax2.tick_params(labelsize=15)

g2.set_xticklabels(state_count.index, rotation = 45)
type_count = pd.DataFrame()

type_count['count'] = stores['type'].value_counts().sort_index()

fig, (ax1,ax2) = plt.subplots(1,2,figsize = (18, 4))

g = sns.barplot(x = type_count.index, y = type_count['count'], ax = ax1)

ax1.set_ylabel('count', fontsize = 15)

ax1.set_xlabel('type',fontsize = 15)

ax.tick_params(labelsize=15)

cluster_count = pd.DataFrame()

cluster_count['count'] = stores['cluster'].value_counts().sort_index()

g = sns.barplot(x = cluster_count.index, y = cluster_count['count'], ax = ax2)

ax2.set_ylabel('count', fontsize = 15)

ax2.set_xlabel('cluster',fontsize = 15)

ax2.tick_params(labelsize=15)
import squarify

fig, ax = plt.subplots(figsize = (16,8))

grouped_city = np.log1p(stores.groupby(['city']).count())

grouped_city['state'] = grouped_city.index

current_palette = sns.color_palette()

squarify.plot(sizes = grouped_city['store_nbr'], label = grouped_city.index, alpha = 0.8,color = current_palette)

plt.rc('font', size = 25)

plt.axis('off')
fig, ax = plt.subplots(figsize = (16,8))

grouped_state = np.log1p(stores.groupby(['state']).count())

grouped_state['state'] = grouped_state.index

current_palette = sns.color_palette()

squarify.plot(sizes = grouped_state['store_nbr'], label = grouped_state.index, alpha = 0.8,color = current_palette)

plt.rc('font', size = 20)

plt.axis('off')
print (items.head())

class_count = pd.DataFrame()

class_count['count'] = items['class'].value_counts().sort_index()

most_freq_class = items['class'].value_counts().head()

perish_count = items['perishable'].value_counts()

family_count = items['family'].value_counts().sort_index()
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize = (18,3))

sns.barplot(x = class_count.index, y = class_count['count'], ax = ax1)

ax1.set_ylabel('count', fontsize = 15)

ax1.set_xlabel('class',fontsize = 15)

ax1.tick_params(labelsize = 6, axis = 'x',which = 'both',top = 'off', bottom = 'off', labelbottom = 'off')

sns.barplot(x = most_freq_class.index, y = most_freq_class.values, ax = ax2)

ax2.set_ylabel('n', fontsize = 15)

ax2.set_xlabel('class-most frequent',fontsize = 15)

ax2.tick_params(labelsize = 12)

sns.barplot(x = perish_count.index, y = perish_count.values, ax = ax3)

ax3.set_ylabel('count', fontsize = 15)

ax3.set_xlabel('perish',fontsize = 15)

ax3.tick_params(labelsize = 12)
fig, ax = plt.subplots(figsize = (18, 3))

g = sns.barplot(x = family_count.index, y = family_count.values, ax = ax)

ax.set_ylabel('count', fontsize = 15)

ax.set_xlabel('family',fontsize = 15)

ax.tick_params(labelsize = 12)

g.set_xticklabels(family_count.index, rotation = 60)
fig, ax = plt.subplots(figsize = (16,8))

grouped_family = np.log1p(items.groupby(['family']).count().head(20))

current_palette = sns.color_palette()

squarify.plot(sizes = grouped_family['class'], label = grouped_family.index, alpha = 0.8,color = current_palette)

plt.rc('font', size = 12)

plt.axis('off')
med_trans = transactions.groupby(['date']).median()

med_trans['date'] = med_trans.index

med_trans['date'] = med_trans['date'].apply(datetime.datetime.strptime, args = ("%Y-%m-%d",))

med_trans.head()
fig, ax = plt.subplots(figsize = (12,6))

sns.set_style('darkgrid')

plt.plot(med_trans['date'], med_trans['transactions'])

#ax = sns.regplot(x='date', y='transactions', data = med_trans)

ax.set_xlabel('date', fontsize=15)

ax.set_ylabel('med_trans', fontsize=15)

ax.tick_params(labelsize=15)
transactions['date'] = transactions['date'].apply(datetime.datetime.strptime, args = ("%Y-%m-%d",))

grouped_trans = transactions.groupby(['date','store_nbr']).sum().unstack()

grouped_trans.head()

fig, ax = plt.subplots(figsize = (12,6))

grouped_trans.plot(ax = ax)
oil['date'] = oil['date'].apply(datetime.datetime.strptime, args = ("%Y-%m-%d",))
import statsmodels.formula.api as sm

regression = (sm.ols(formula="dcoilwtico ~ date ", data=oil).fit())
oil['trend'] = regression.predict(oil['date'])
fig, ax = plt.subplots(figsize = (18, 5))

ax.plot(oil['date'],oil['dcoilwtico'], label = 'oilprice')

ax.set

ax.set_xlabel('date', fontsize=15)

ax.set_ylabel('oilprice', fontsize=15)

ax.tick_params(labelsize=15)
lag7 = oil['dcoilwtico'] - oil['dcoilwtico'].shift(7)
fig, ax = plt.subplots(figsize = (18, 5))

ax.plot(oil['date'],lag7, label = 'weekly variations in oil price')

ax.set

ax.set_xlabel('date', fontsize=15)

ax.set_ylabel('weekly variations', fontsize=15)

ax.tick_params(labelsize=15)
holiday_events.head()
holiday_events['date'] = holiday_events['date'].apply(datetime.datetime.strptime, args = ("%Y-%m-%d",))
type_count = holiday_events['type'].value_counts().sort_index()

locale_count = holiday_events['locale'].value_counts().sort_index()
fig, (ax1, ax2) = plt.subplots(1,2,figsize = (18, 4))

sns.barplot(x = type_count.index, y = type_count.values, ax = ax1)

ax1.set_ylabel('count', fontsize = 15)

ax1.set_xlabel('type',fontsize = 15)

sns.barplot(x = locale_count.index, y = locale_count.values, ax = ax2)

ax2.set_ylabel('count', fontsize = 15)

ax2.set_xlabel('locale',fontsize = 15)
most_freq_descr = holiday_events['description'].value_counts().head(12).sort_index()

#print (most_freq_descr)

transferred_count = holiday_events['transferred'].value_counts()
fig, (ax1, ax2) = plt.subplots(1,2,figsize = (18, 4))

sns.barplot(y = most_freq_descr.index, x = most_freq_descr.values, ax = ax1)

ax1.set_ylabel('Description-most frequent', fontsize = 15)

ax1.set_xlabel('Frequency',fontsize = 15)

sns.barplot(x = transferred_count.index, y = transferred_count.values, ax = ax2)

ax2.set_ylabel('count', fontsize = 15)

ax2.set_xlabel('transferred',fontsize = 15)
locale_name_count = holiday_events['locale_name'].value_counts().sort_index()

locale_name_count.head()

fig, ax = plt.subplots(figsize = (18, 3))

g = sns.barplot(x = locale_name_count.index, y = locale_name_count.values, ax = ax)

ax.set_ylabel('count', fontsize = 15)

ax.set_xlabel('locale_name',fontsize = 15)

ax.tick_params(labelsize = 12)

g.set_xticklabels(locale_name_count.index, rotation = 45)