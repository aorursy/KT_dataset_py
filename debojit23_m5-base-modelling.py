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
import gc

import os

from pathlib import Path

import random

import sys





from ipywidgets import interact, interactive, fixed, interact_manual

from ipywidgets import widgets

from IPython.display import display

from tkinter import *





from tqdm.notebook import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff



# --- models ---

from sklearn import preprocessing



from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



# --- setup ---

pd.set_option('max_columns', 50)





#macro

month_arr=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# get pandas dataframes

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sales_train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

submission
d = ['d_' + str(i) for i in range(1802,1914)]

sales_train_mlt = pd.melt(sales_train, id_vars=['item_id','dept_id','cat_id','store_id','state_id'], value_vars=d)

sales_train_mlt = sales_train_mlt.rename(columns={'variable':'d', 'value':'sales'})



sales_train_mlt=pd.merge(sales_train_mlt,calendar,how='inner',on=['d'])







sell_prices=sell_prices.merge(sales_train_mlt,on=['wm_yr_wk','store_id','item_id'],how='inner')

#sell_prices['net_selling_price']=sell_prices['sales']*sell_prices['sell_price']
sell_prices
sales_train_mlt=sales_train_mlt.fillna(-1)

sales_train_event2=sales_train_mlt[sales_train_mlt['event_name_2']!=-1 ]

sales_train_event1=sales_train_mlt[sales_train_mlt['event_name_1']!=-1 ]

sales_train_event1
#Lineplot analysis for every month (SELL PRICE)

features2_plot=['sales','sell_price','net_selling_price']



sell_prices['date']=pd.to_datetime(sell_prices['date'])

sell_prices.sell_price.astype('int32')

print("change the value of q for changing the month \n change the value of p to select the feature")

def f(p,q):

  temp_df=pd.DataFrame()

  #temp_df=sell_prices['sales']

  temp_df1=sell_prices[sell_prices['month']==q]

  #print(temp_df1)

  v=['date',features2_plot[p-1]]

  print("Feature Selected : ",features2_plot[p-1])

  label='Avg '+features2_plot[p-1]+' by day'



  temp_df1[v].set_index('date').resample("D")['sales'].mean().plot(kind='line',figsize=(12,10),grid=True)

  plt.xlabel("Days")

  plt.ylabel(label)

    

interact(f,p=widgets.IntSlider(min=1,max=2,step=1),q=widgets.IntSlider(min=1,max=6,step=1))



sell_prices[['date','sales']].set_index('date').resample("D")['sales'].mean().plot(kind='line',figsize=(12,10),grid=True,color='r')

plt.title("Average sales accross 6 months")

departments=sell_prices.dept_id.unique()

categories=sell_prices.cat_id.unique()

cities=['California','Texas','Winsconsin']

locations=sell_prices.state_id.unique()

stores=sell_prices.store_id.unique()

features2_plot=['dept_id','cat_id','state_id']

print("change the value of loc for shuffling locations  \nchange the value of cat to toggle between the categories")

def f(loc,cat,dept):

  temp_df=sell_prices[sell_prices['state_id']==locations[loc]]

  temp_df1=temp_df[temp_df['cat_id']==categories[cat]]

  temp_df2=temp_df1[temp_df1['dept_id']==departments[dept]]



  print(temp_df.shape,'   location selected : ',locations[loc])

  print(temp_df1.shape,'    categories selected : ',categories[cat])

  print(temp_df2.shape,' department selected : ',departments[dept])

  label="Selling price across all "+cities[loc]+" under "+categories[cat]+" category "+" and the department is " + departments[dept]

  fig = plt.figure(figsize=(16,12))

  ax = sns.lineplot(y="sales", x="d", data=temp_df2)

  ax.set_title(label)

  ax.set_xlabel('days(d_1802-d_1913)')

  ax.set_ylabel('Sales ')

    

interact(f,loc=widgets.IntSlider(min=0,max=len(locations)-1,step=1),cat=widgets.IntSlider(min=0,max=len(categories)-1,step=1),dept=widgets.IntSlider(min=0,max=len(departments)-1,step=1))

#distribution plot analysis

#1 sales distribution vs all stores in different locations

#2 sales distribution vs all dept

features2_plot=['dept_id','cat_id','state_id']

stores

print("To use the toggle bar(loc/cat/dept) set the stor to -1 and vice-versa")





def f(loc,cat,dept,stor):

  temp_df=pd.DataFrame()

  temp_df=sell_prices[sell_prices['state_id']==locations[loc]]

  temp_df1=temp_df[temp_df['cat_id']==categories[cat]]

  temp_df2=temp_df1[temp_df1['dept_id']==departments[dept]]

  temp_df3=sell_prices[sell_prices['store_id']==stores[stor]]



  fig = plt.figure(figsize=(12,8))



  if stor>-1:

    print("Store Selected ",stores[stor])

    label="Sales distributions of Store "+stores[stor]

    sns.distplot(temp_df3['sales'],rug=False, hist=True,kde=True,kde_kws={"color":"black"},hist_kws={"color":"red"})

    plt.title(label)

  else:

    print('location selected : ',locations[loc])

    print('categories selected : ',categories[cat])

    print(' department selected : ',departments[dept])

    label="Sales distributions across all "+locations[loc]+" under "+categories[cat]+" category "+" and the department is " + departments[dept]

    sns.distplot(temp_df2['sales'],rug=False, hist=True,kde=True,kde_kws={"color":"black"},hist_kws={"color":"red"})

    plt.title(label)





interact(f,loc=widgets.IntSlider(min=0,max=len(locations)-1,step=1),

         cat=widgets.IntSlider(min=0,max=len(categories)-1,step=1),

         dept=widgets.IntSlider(min=0,max=len(departments)-1,step=1),

         stor=widgets.IntSlider(min=-1,max=len(stores)-1))

import plotly.express as px

fig = px.sunburst(sell_prices, path=['state_id', 'store_id', 'cat_id'], values='sales')

fig.show()
import plotly.express as px

fig = px.sunburst(sell_prices, path=[ 'store_id', 'cat_id'], values='sales')

fig.show()
sell_prices.nunique()
sell_prices[sell_prices.weekday=='Sunday']