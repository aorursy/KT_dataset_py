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

root = '/kaggle/input/competitive-data-science-predict-future-sales/'

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv(os.path.join(root, 'sales_train.csv'))

train_df

test_df = pd.read_csv(os.path.join(root, 'test.csv'))

test_df
for index, ele in enumerate(train_df['item_cnt_day']):

    if (ele.is_integer() == False) | (ele < 0) :

        train_df.at[index, 'item_cnt_day']  = np.NaN

train_df.dropna(inplace = True)

train_df = train_df.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day':np.sum}).reset_index()

train_df
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#plt.scatter(x = train_df['shop_id'], y = train_df['item_cnt_day'])
plt.scatter(x = train_df['date_block_num'], y = train_df['item_cnt_day'])
train_df.at[train_df['item_cnt_day'] > 500, 'item_cnt_day'] = np.NaN

train_df.dropna(inplace = True)

#plt.scatter(x = train_df['item_id'], y = train_df['item_cnt_day'])

train_df.describe()
from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors = 10, weights= 'distance')

from  sklearn.ensemble import RandomForestClassifier

#rfc = RandomForestClassifier(n_estimators = 10, max_depth = 20)



from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimator = 100, early_stopping_rounds = 5, learning_rate = 0.05)



train_df_copy = train_df

train_df_copy = train_df_copy.drop(labels = ['date_block_num'], axis = 1)

train_y = train_df_copy.pop('item_cnt_day')

train_df_copy.describe()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

clf_train = scaler.fit_transform(train_df_copy)

xgb.fit(clf_train,train_y)



from sklearn.model_selection import cross_val_score

scores = cross_val_score(xgb, clf_train, train_y, cv=2)

scores
test_df_copy = test_df.copy()

test_id = test_df_copy.pop('ID')

clf_test = scaler.transform(test_df_copy)

test_y = xgb.predict(clf_test)

test_y = test_y.T

res = pd.DataFrame(test_y, columns = ['item_cnt_month'])

res.to_csv('salescount.csv',index=True)