# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
#importing packages that we used for forecasting Future Sales

import warnings

warnings.filterwarnings("ignore")

import itertools

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

import statsmodels.tsa.api as smt

%matplotlib inline

import seaborn as sns

import datetime

!pip install wordcloud
#Train Data

train.head()
#Test Data

test.head()
#Item Categories Data

item_categories.head()
#Item Data

items.head()
#Shop Data

shops.head()
#Sample Submission Data (i.e., we want ID and item_cnt_month in our output)

sample_submission.head()
train.head(5).append(train.tail(5))
#Checking the data types of the variables and here date is object type and we have to convert it into date type

dict(train.dtypes)
#Here we are converting data type of date from object to date

train['date'] = pd.to_datetime(train['date'],format='%d.%m.%Y')
#Checking the info of Train data

train.describe().T
test.shape
train.head()
#Checking outliers 



for c in ['item_cnt_day','item_price']:

    plt.figure()

    plt.title(c)

    sns.boxplot(train[c])
#Handling Outliers



train = train[train['item_cnt_day']<=1000]

train = train[train['item_price']<100000]
#there is this one column in the train data where item price is less than 0 which is not possible so I am replacing the value with median.



train[train.item_price < 0]



median = train[(train.date_block_num==4)&(train.shop_id==32)&(train.item_id==2973)&(train.item_cnt_day==1.0)&(train.item_price>0)].item_price.median()

train.loc[train.item_price<0,'item_price'] = median

#We can find that there are values in item_cnt_day (item sold per day) less than 0 so lets replace that values with 0

train[train.item_cnt_day<0]

train.loc[train.item_cnt_day<0,'item_cnt_day'] = 0
train.isnull().sum()
#The objective is to predict the sales for the next month (no. of items sold next month i.e., Nov 2015)

#And to predict the sales for nov 2015 month lets first compute the total sales per month.



sales_ts = train.groupby(['date_block_num'])['item_cnt_day'].sum()

sales_ts
plt.figure(figsize=(10,5))

plt.plot(sales_ts)

plt.xlabel("month")

plt.ylabel("Sale")

plt.show()
fig, axes = plt.subplots(2,2)

fig.set_figwidth(14)

fig.set_figheight(8)



axes[0][0].plot(sales_ts.index, sales_ts, label='original')

axes[0][0].plot(sales_ts.index, sales_ts.rolling(window=4).mean(), label = '4-months rolling window')

axes[0][0].set_xlabel("Months")

axes[0][0].set_ylabel("Sales")

axes[0][0].set_title("$-Months Moving average")

axes[0][0].legend(loc='best')





axes[0][1].plot(sales_ts.index,sales_ts,label='original')

axes[0][1].plot(sales_ts.index, sales_ts.rolling(window=8).mean(), label="8-months rolling window")

axes[0][1].set_xlabel("months")

axes[0][1].set_ylabel("Sales")

axes[0][1].set_title("8-Months moving average")

axes[0][1].legend(loc='best')



axes[1][0].plot(sales_ts.index,sales_ts,label="original")

axes[1][0].plot(sales_ts.index,sales_ts.rolling(window=10).mean(), label='10-months rolling window')

axes[1][0].set_xlabel("months")

axes[1][0].set_ylabel("sales")

axes[1][0].set_title("10-Months Moving Average")

axes[1][0].legend(loc='best')



axes[1][1].plot(sales_ts.index, sales_ts, label='original')

axes[1][1].plot(sales_ts.index, sales_ts.rolling(window=12).mean(), label="12-months rolling window")

axes[1][1].set_xlabel("months")

axes[1][1].set_ylabel("12-months moving average")

axes[1][1].legend(loc='best')
monthly_sales_data = pd.pivot_table(train,values="item_cnt_day",index=['shop_id','item_id'],columns='date_block_num',

                                   aggfunc='sum',fill_value=0)

monthly_sales_data.reset_index(inplace=True)

monthly_sales_data
monthly_sales_data.plot()
decomposition = sm.tsa.seasonal_decompose(sales_ts,period=12,model='multiplicative')

fig = decomposition.plot()

fig.set_figwidth(12)

fig.set_figheight(8)

fig.suptitle("Decomposition of multiplicative time series")

plt.show()
decomp_output = pd.DataFrame(pd.concat([decomposition.observed,decomposition.trend,decomposition.seasonal,decomposition.resid],axis=1))

decomp_output

decomp_output['TSI'] = decomp_output.trend*decomp_output.seasonal*decomp_output.resid

decomp_output
!pip install fbprophet  #installing fbprophet using pip install
from fbprophet import Prophet
ts = sales_ts.copy()

ts.index = pd.date_range(start='2013-01-01',end='2015-10-01',freq = 'MS')

ts = ts.reset_index()

ts.head()

ts.columns=['ds','y']

ts
sales_model = Prophet(interval_width=0.95,seasonality_mode = 'multiplicative')

sales_model.fit(ts)
sales_forecast = sales_model.make_future_dataframe(periods=2,freq='MS')

sales_forecast = sales_model.predict(sales_forecast)

sales_forecast.tail()
plt.figure(figsize=(10,10))

sales_model.plot(sales_forecast)

plt.xlabel('months')

plt.ylabel('sales')
sales_model.plot_components(sales_forecast)
x = items.item_category_id.nunique()

print("number of unique item category = " +str(x))
#Making a wordcloud for item category name.

from wordcloud import STOPWORDS

from wordcloud import WordCloud



stopwords = set(STOPWORDS)

plt.figure(figsize=(12,10))



wordcloud = WordCloud(max_font_size=100,width=2000,height=1000).generate(str(items.item_name))

plt.title("Wordcloud of item names",fontsize=30)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
plt.figure(figsize=(20,5))

sns.barplot('item_category_id','item_id',data=items)
x= shops.shop_id.nunique()

print("Number of unique shops = "+str(x))
#Wordcloud for shop names

from wordcloud import WordCloud

from wordcloud import STOPWORDS



stopwords = set(STOPWORDS)

plt.figure(figsize=(12,10))

wordcloud = WordCloud().generate(str(shops.shop_name))

plt.title("Wordcloud of Shop names",fontsize=30)

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()
item_categories.head(2)
x = item_categories.item_category_id.nunique()

print("unique item_categories ID = "+str(x))
#Wordcloud of item category names

from wordcloud import WordCloud

from wordcloud import STOPWORDS



stopwords = set(STOPWORDS)

plt.figure(figsize=(12,10))

wordcloud = WordCloud().generate(str(item_categories.item_category_name))

plt.imshow(wordcloud, interpolation='bilinear')

plt.title("Wordcloud of Item category Names",fontsize=30)

plt.axis("off")

plt.show()
train['year'] = train.date.dt.strftime('%Y')

train['month'] = train.date.dt.strftime('%m')

train['day'] = train.date.dt.strftime('%d')
train.head()
#lets check which year, months and days are the busiest

for x in ['year','month','day']:

    plt.figure(figsize=(10,5))

    sns.countplot(x,data=train)

    plt.show()
train.tail()
#month wise data

monthly_data = train.groupby(['date_block_num','shop_id','item_id']).sum().reset_index()

monthly_data = monthly_data[['date_block_num','shop_id','item_id','item_cnt_day']]

monthly_data.head()
#lets create Pivot table represent monthwise data

data = monthly_data.pivot_table(values='item_cnt_day',index=['shop_id','item_id'],columns='date_block_num',fill_value=0)

data.reset_index()
data = pd.merge(test, data, on=['shop_id','item_id'],how='left')

data.fillna(0,inplace=True)

data
data.drop(['shop_id','item_id','ID'],axis=1,inplace=True)

data
# for train we will keep all the columns except last one

train_x = data.values[:,:-1]

train_x.shape

train_y = data.values[:,-1:]

train_y.shape
# for test we will keep all the columns except the first one

test_x = data.values[:,1:]

test_x.shape
from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from sklearn.model_selection import KFold, GridSearchCV, train_test_split
"""

params = {

    'max_depth': [6,8,10,12],

    'n_estimators': [50,100,200,300],

    'learning_rate': [0.01,0.1,0.02,0.2]

}

kfold = KFold(n_splits=5,shuffle=True,random_state=1)

gscv = GridSearchCV(XGBRegressor(),param_grid=params, verbose=1, cv=kfold, n_jobs=-1)

gscv.fit(train_x, train_y)

print(gscv.best_score_)

print(gscv.best_params_)

"""
#Modelling

#get the test set prediction and clip value to the specified range

xgbr  = XGBRegressor(learning_rate=0.01,max_depth=8,n_estimators=100)

xgbr.fit(train_x,train_y)

pred_y = xgbr.predict(test_x).clip(0.,20.)
Submission = pd.DataFrame(pred_y,columns=['item_cnt_month'])

Submission.to_csv("submission.csv",index_label='ID')