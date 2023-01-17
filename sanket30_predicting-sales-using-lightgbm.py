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
# First we create a dataframe with the raw sales data, which we'll reformat later
DATA = '../input/'
sales = pd.read_csv(DATA+'sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
sales.head()
# Let's also get the test data
test = pd.read_csv(DATA+'test.csv')
test.head()
# Now we convert the raw sales data to monthly sales, broken out by item & shop
# This placeholder dataframe will be used later to create the actual training set
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
df = df[['date','item_id','shop_id','item_cnt_day']]
df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()
df.head()
# Merge the monthly sales data to the test data
# This placeholder dataframe now looks similar in format to our training data
df_test = pd.merge(test, df, on=['item_id','shop_id'], how='left')
df_test = df_test.fillna(0)
df_test.head()
# Remove the categorical data from our test data, we're not using it
df_test = df_test.drop(labels=['ID', 'shop_id', 'item_id'], axis=1)
df_test.head()
# Now we finally create the actual training set
# Let's use the '2015-10' sales column as the target to predict
TARGET = '2015-10'
y_train = df_test[TARGET]
X_train = df_test.drop(labels=[TARGET], axis=1)

print(y_train.shape)
print(X_train.shape)
X_train.head()
# To make the training set friendly for keras, we convert it to a numpy matrix
# X_train = X_train.as_matrix()
# X_train = X_train.reshape((214200, 33, 1))

# y_train = y_train.as_matrix()
# y_train = y_train.reshape(214200, 1)

print(y_train.shape)
print(X_train.shape)

# X_train[:1]
# Lastly we create the test set by converting the test data to a numpy matrix
# We drop the first month so that our trained LSTM can output predictions beyond the known time range
X_test = df_test.drop(labels=['2013-01'],axis=1)
# X_test = X_test.as_matrix()
# X_test = X_test.reshape((214200, 33, 1))
print(X_test.shape)
from lightgbm import LGBMRegressor
model=LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=32,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        max_depth=8,
        reg_alpha=0.04,
        reg_lambda=0.073,
        min_split_gain=0.0222415,
        min_child_weight=40)


print('Training time, it is...')
model.fit(X_train, y_train,
          
         )
# Get the test set predictions and clip values to the specified range
y_pred = model.predict(X_test).clip(0., 20.)

# Create the submission file and submit!
preds = pd.DataFrame(y_pred, columns=['item_cnt_month'])
preds.to_csv('submission.csv',index_label='ID')
