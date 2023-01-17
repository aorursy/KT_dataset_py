# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
demand = pd.read_csv("../input/train.csv", sep= ",",encoding = 'utf-8')
meal = pd.read_csv("../input/meal_info.csv", sep= ",",encoding = 'utf-8')
fulfillment = pd.read_csv("../input/fulfilment_center_info.csv", sep= ",",encoding = 'utf-8')
test = pd.read_csv("../input/test_QoiMO9B.csv", sep= ",",encoding = 'utf-8')
demand_combine1 = pd.merge(demand, meal, on='meal_id', how='left')
train = pd.merge(demand_combine1, fulfillment, on='center_id', how='left')
test_combine1 = pd.merge(test, meal, on='meal_id', how='left')
test_update = pd.merge(test_combine1, fulfillment, on='center_id', how='left')
train['discount']=train['base_price']-train['checkout_price']
train['discount_percentage']=(train['discount']/train['base_price'])*100
test_update['discount']=test_update['base_price']-test['checkout_price']
test_update['discount_percentage']=(test_update['discount']/test_update['base_price'])*100
train_dummies = pd.get_dummies(train)
test_update_dummies = pd.get_dummies(test_update)
Xtest = test_update_dummies[['id', 'week', 'center_id', 'meal_id', 'checkout_price', 'base_price','emailer_for_promotion', 'homepage_featured', 'city_code','region_code', 'op_area','discount_percentage','category_Beverages', 'category_Biryani', 'category_Desert','category_Extras', 'category_Fish', 'category_Other Snacks','category_Pasta', 'category_Pizza', 'category_Rice Bowl', 'category_Salad', 'category_Sandwich', 'category_Seafood','category_Soup', 'category_Starters', 'cuisine_Continental','cuisine_Indian', 'cuisine_Italian', 'cuisine_Thai','center_type_TYPE_A', 'center_type_TYPE_B', 'center_type_TYPE_C']]
import xgboost as xgb
xgb_reg = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, gamma=0, max_depth=13)
from sklearn.model_selection import train_test_split
Y = train_dummies['num_orders']
X = train_dummies[['id', 'week', 'center_id', 'meal_id', 'checkout_price', 'base_price','emailer_for_promotion', 'homepage_featured', 'city_code','region_code', 'op_area','discount_percentage','category_Beverages', 'category_Biryani', 'category_Desert','category_Extras', 'category_Fish', 'category_Other Snacks','category_Pasta', 'category_Pizza', 'category_Rice Bowl', 'category_Salad', 'category_Sandwich', 'category_Seafood','category_Soup', 'category_Starters', 'cuisine_Continental','cuisine_Indian', 'cuisine_Italian', 'cuisine_Thai','center_type_TYPE_A', 'center_type_TYPE_B', 'center_type_TYPE_C']]

xgb_reg.fit(X,Y)
test_pred=xgb_reg.predict(Xtest)
for i in range(len(test_pred)):
    if test_pred[i]<=0:
        test_pred[i]=1
test_update_dummies['num_orders'] = test_pred
submission = test_update_dummies[['id','num_orders']]
submission.to_csv("../Submission.csv",sep = ",",encoding = 'utf-8',index = False)
