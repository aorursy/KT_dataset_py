# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotnine 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

sell = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

sub = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")
print("Unit sales of all products, aggregated for each state", train['state_id'].nunique())

print("Unit sales of all products, aggregated for each store", train['store_id'].nunique())

print("Unit sales of all products, aggregated for each category", train['cat_id'].nunique())

print("Unit sales of all products, aggregated for each department", train['dept_id'].nunique())

print("Unit sales of all products, aggregated for each State and category", train['state_id'].nunique() * train['cat_id'].nunique())

print("Unit sales of all products, aggregated for each State and department", train['state_id'].nunique() * train['dept_id'].nunique())

print("Unit sales of all products, aggregated for each store and category", train['store_id'].nunique() * train['cat_id'].nunique())

print("Unit sales of all products, aggregated for each store and department", train['store_id'].nunique() * train['dept_id'].nunique())

print("Unit sales of all products, aggregated for each  and category", train['dept_id'].nunique() * train['cat_id'].nunique())

print("Unit sales of product x, aggregated for all stores/states", train['item_id'].nunique())

print("Unit sales of product x, aggregated for all states", train['item_id'].nunique() * train['state_id'].nunique())

print("Unit sales of product x, aggregated for all stores", train['item_id'].nunique() * train['store_id'].nunique())
calendar.head(8)
calendar[calendar['event_name_1'].notnull()].head()
from plotnine import *

import plotnine
agg = calendar.groupby('event_name_1')['event_name_1'].agg({'count'}).reset_index()

(ggplot(data = agg) 

  + geom_bar(aes(x='event_name_1', y='count'), fill='#49beb7', color='black', stat='identity')

  + scale_color_hue(l=0.45)

  + theme_light() 

  + theme(

         axis_text_x = element_text(angle=80),

         figure_size=(12,8),

         legend_position="none"))
agg = calendar.groupby('event_type_1')['event_type_1'].agg({'count'}).reset_index()

(ggplot(data = agg) 

  + geom_bar(aes(x='event_type_1', y='count'), fill='#49beb7', color='black', stat='identity')

  + scale_color_hue(l=0.45)

  + theme_light() 

  + theme(

         axis_text_x = element_text(angle=80),

         figure_size=(12,8),

         legend_position="none"))
calendar[calendar['event_name_2'].notnull()].head()
print("event_name_2 notnull shape : ", calendar[calendar['event_name_2'].notnull()].shape)

print("event_name_1 and 2 notnull shape : ", calendar[(calendar['event_name_2'].notnull()) & (calendar['event_name_1'].notnull())].shape)
agg = calendar.groupby('event_name_2')['event_name_2'].agg({'count'}).reset_index()

(ggplot(data = agg) 

  + geom_bar(aes(x='event_name_2', y='count'), fill='#49beb7', color='black', stat='identity')

  + scale_color_hue(l=0.45)

  + theme_light() 

  + theme(

         axis_text_x = element_text(angle=80),

         figure_size=(12,8),

         legend_position="none"))
agg = calendar.groupby('event_type_2')['event_type_2'].agg({'count'}).reset_index()

(ggplot(data = agg) 

  + geom_bar(aes(x='event_type_2', y='count'), fill='#49beb7', color='black', stat='identity')

  + scale_color_hue(l=0.45)

  + theme_light() 

  + theme(

         axis_text_x = element_text(angle=80),

         figure_size=(12,8),

         legend_position="none"))
print(sell.shape)

sell.head()
print(train.shape)

train.head()
sub.head()