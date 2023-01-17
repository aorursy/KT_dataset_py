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
train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv") 

test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv") 

shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

Sample = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

item_cat = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
Sample.head()
test.head()
train.head()
#no null values in the train data

train.isnull().sum()
train.shape
#replacing all the item count of -1 to 0

train['item_cnt_day']=train['item_cnt_day'].replace(-1,0)
train.head()
#data of 0 sales items on a particular day

train[train['item_cnt_day']==0]
train['date'].min(), train['date'].max()
import datetime

#formatting the date column correctly

train.date=train.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

# check

print(train.info())
import seaborn as sns

# number of items per cat 

x=items.groupby(['item_category_id']).count()

x=x.sort_values(by='item_id',ascending=False)

x=x.iloc[0:10].reset_index()

x

# #plot

plt.figure(figsize=(8,4))

ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)

plt.title("Items per Category")

plt.ylabel('number of items', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
#compute the total sales per month and plot that data.

ts=train.groupby(["date"])["item_cnt_day"].sum()

ts.astype('float')

plt.figure(figsize=(16,8))

plt.title('Total Sales of the company')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(ts)
#training the model based on the['shop_id', 'item_id']

X = train.iloc[:,2:4]

Y = train.iloc[:,5]
X
#Train a model using XGBoost

import xgboost as xgb

model = xgb.XGBRegressor(max_depth = 10, min_child_weight=0.5, subsample = 1, eta = 0.3, num_round = 1000, seed = 1)

model.fit(X, Y)
test1 = test.drop('ID',axis=1)
test1
#predicting on the test data

Predict=model.predict(test1)
Predict = Predict.astype(int)

print(Predict)
Submission = pd.DataFrame(data=Predict,columns=['item_cnt_month'])

Submission['ID'] = test['ID']

Submission = Submission[['ID','item_cnt_month']]
Submission
Submission.to_csv('submission.csv',index=False)