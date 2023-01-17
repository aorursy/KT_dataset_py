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
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew 


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
print(train.shape)
print(test.shape)
print(items.shape)
print(item_categories.shape)
print(shops.shape)
train.head()
test.head()
items.head()
item_categories.head()
shops.head()
test.info()
train.info()
train.item_cnt_day.plot()
plt.title("Number of products sold per day");
def graph_insight(data):
    print(set(data.dtypes.tolist()))
    df_num = data.select_dtypes(include = ['float64', 'int64'])
    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);
    
graph_insight(train)
#We are trying to predict the sale price column
target = train.item_cnt_day

#Get rid of the answer and anything thats not an object
train = train.drop(['item_price','item_cnt_day','date','date_block_num'],axis=1).select_dtypes(exclude=['object'])

#Split the data into test and validation
train_X, test_X, train_y, test_y = train_test_split(train,target,test_size=0.25)
#train_X.fillna( method ='ffill', inplace = True)
#test_X.fillna( method ='ffill', inplace = True)


my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=8, 
             eval_set=[(test_X, test_y)], verbose=False)


#Make predictions
predictions = my_model.predict(test_X)

print("Mean absolute error = " + str(mean_absolute_error(predictions,test_y)))

#Getting it to the right format that we used with our model
test = test.select_dtypes(exclude=['object'])
test.fillna( method ='ffill', inplace = True)
#Fill in all the NaN values with ints
test_X = test
test_X = test_X.drop(['ID'],axis=1).select_dtypes(exclude=['object'])

#Make predictions
predictions = my_model.predict(test_X)

my_submission = pd.DataFrame({'Id': test.ID, 'item_cnt_month': predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submi2.csv', index=False)