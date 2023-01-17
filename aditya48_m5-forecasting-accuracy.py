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
## Importing the Libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

import plotly.graph_objects as go

## Loading the Dataset:



train_x = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")

calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

price = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

train_x.head(10)
train_x.dtypes
train_x.isnull().sum()
train_x['cat_id'].value_counts()
train_x['dept_id'].value_counts()
train_x['store_id'].value_counts()
train_x['state_id'].value_counts()
train_x.columns
calendar.head(10)
calendar.shape
calendar.dtypes
calendar.columns
## Event_name_1: is basically an event occuring on that particular day. 

calendar['event_name_1'].value_counts()
## Event_name_2 : Only 5 values are present and further there will be NAN values.

## Not sure whether it is useful or not.

calendar['event_name_2'].value_counts()
train_x.columns


train_x['Units'] = train_x.sum(axis=1, skipna=True)

train_x.head()
data = pd.concat([train_x,calendar],axis=1)

data.head()
price.head()
print(price.isnull().sum())

print()

print(price.shape)
# Let's see the price variation.

sns.boxplot(price['store_id'],price['sell_price']).set_title("SELLING PRICE DISTRIBUTION FOR DIFFERENT STORE")
## Let's see these Outliers:

temp=price[price['sell_price']>100]

print(temp.shape)

print()

print(temp)
temp = train_x['Units'].head(100)

sns.boxplot(temp).set_title("DISTRIBUTION OF UNITS SOLD")
plt.bar(x=train_x['cat_id'],height=train_x['Units'])

plt.xlabel("CATEGORIES")

plt.ylabel("UNITS SOLD")

plt.title("UNITS SOLD CATEGORY WISE")

plt.show()
plt.bar(x=train_x['state_id'],height=train_x['Units'])

plt.xlabel("STATES")

plt.ylabel("UNITS SOLD")

plt.title("UNITS SOLD STATES WISE")

plt.show()
sns.barplot(x='cat_id',y='Units',

           hue='state_id',

           data=train_x).set_title("UNITS SOLD IN STATES IN DIFFERENT CATEGORIES")
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

sns.barplot(x='dept_id',y='Units',

           hue='state_id',

           data=train_x,ax=ax).set_title("UNITS SOLD IN STATES IN DIFFERENT DEPARTMENTS")







## ax : matplotlib Axes, optional

## Axes object to draw the plot onto, otherwise uses the current Axes.

## PROCESS:

## Give Dimensions:

## use plt.subplots and then put ax inside sns.barplot.