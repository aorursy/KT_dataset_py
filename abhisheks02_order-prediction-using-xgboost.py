# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_orders                  = pd.read_csv('../input/instacart-market-basket-analysis/orders.csv')

df_order_products_train    = pd.read_csv('../input/instacart-market-basket-analysis/order_products__train.csv')

df_order_products_prior    = pd.read_csv('../input/instacart-market-basket-analysis/order_products__prior.csv')

df_products                = pd.read_csv('../input/instacart-market-basket-analysis/products.csv')

df_departments             = pd.read_csv('../input/instacart-market-basket-analysis/departments.csv')

df_aisles                  = pd.read_csv('../input/instacart-market-basket-analysis/aisles.csv')

df_sample_submission       = pd.read_csv('../input/instacart-market-basket-analysis/sample_submission.csv')
df_orders_products = pd.merge(df_order_products_train , df_orders, on = 'order_id')
df_orders_products.dropna()
X = df_orders_products[['user_id','product_id','order_id',

                        'add_to_cart_order','order_number','order_dow','order_hour_of_day']]

y = df_orders_products.reordered
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
xgb = XGBClassifier() 



param_grid = {

        

        'max_depth': [1,2,3,4],

        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],

        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],

        'gamma': [0, 0.25, 0.5, 1.0],

        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],

        'n_estimators': [100]}

clf = RandomizedSearchCV(xgb, param_grid, n_iter=10,

                            n_jobs= -1, verbose=2, cv=3, random_state=1)
clf.fit(X_train, y_train)  

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test,y_pred) 



print(acc)
f1_score(y_test,y_pred)