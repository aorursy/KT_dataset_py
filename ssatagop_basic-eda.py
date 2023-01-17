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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as ply

import plotly.express as px
import pandas as pd

calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

train = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")

sample_submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")

sell_prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
train.head(10)
#selecting just the numerical features to do basic calculation

num_feats =  train.select_dtypes(exclude = ["object"]).columns

num_feats
train['item_sum']= train[num_feats].sum(axis=1) # calculating the total sales in each of the month against each item_id

train.head(15)
# combine validation data set and calendar data set

train_cal = pd.concat([train,calendar], axis =1)

train_cal.head(10)
fig1 = sns.boxplot(x=train_cal['store_id'], y = train_cal['item_sum'], showfliers = False)

#average sales across items and stores by category. Again data is noisy. 

train['mean_sales']=train[num_feats].mean(axis=1)

train_cal['mean_sales']=train['mean_sales']

plt.ylim(0, 40)

plt.figure(figsize=(15,8))

fig6 = sns.boxplot(x = train_cal['cat_id'], y = train_cal['mean_sales'], hue = train_cal['state_id'], showfliers = False)
#df = px.data.train_cal()

fig = px.scatter_matrix(train_cal,

    dimensions=['mean_sales','event_name_1','event_name_2','state_id'],

    color="cat_id")

fig.show()
plt.figure(figsize=(15,8))

fig6 = sns.boxplot(x = train_cal['year'], y = train_cal['mean_sales'], hue = train_cal['state_id'], showfliers = False)
fig5 = px.pie(train_cal, values='mean_sales', names='state_id') #average sales by state

fig5.show()
fig6 = px.pie(train_cal, values='mean_sales', names='cat_id') # Average Sales per category

fig6.show()
fig6 = px.pie(train_cal, values='mean_sales', names='store_id')

fig6.show()
naive_avg = train.set_index('id')[num_feats[-30:]].mean(axis=1).to_dict()

fcols = [f for f in sample.columns if 'F' in f]

#fcols = train.columns

for f in fcols:

#count = len(train.columns)

#count

#for i in train.columns:

    sample[i] = sample['id'].map(naive_avg).fillna(0)

    

sample.to_csv('M5_submission.csv', index=False)

print("Your submission is saved")