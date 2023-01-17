# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from xgboost.sklearn import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

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





#df_orders                  = pd.read_csv('../input/instacart-market-basket-analysis/orders.csv')

#print(df_orders.shape)

#train = df_orders.loc[(df_orders.eval_set == 'train')]

#print(train.shape)

#test = df_orders.loc[(df_orders.eval_set == 'test')]

#print(test.shape)

#prior = df_orders.loc[(df_orders.eval_set == 'prior')]

#print(prior.shape)

#print(df_orders.head(100))

#products = pd.merge(df_products, df_aisles, on='aisle_id', how='inner')

#products = pd.merge(products, df_departments , on='department_id', how='inner')

#print(products.head())

#products.shape

#products.isnull().any()
#df_order_products_training = pd.concat([df_order_products_train,df_order_products_prior],ignore_index=True)

#print(df_order_products_prior.shape)

#print(df_order_products_train.shape)
#df_order_products_p_t_t = pd.merge(df_order_products_training , products, on = 'product_id')

#print(df_order_products_p_t_t.head())

#print(df_order_products_p_t_t.shape)

df_orders_products = pd.merge(df_order_products_train , df_orders, on = 'order_id')

df_orders_products.head()
df_orders_products.dropna()

#df_orders_products.shape
#train = df_orders_products.loc[(df_orders_products.eval_set == 'train')]

#print(train.shape)
#test = df_orders_products.loc[(df_orders_products.eval_set == 'prior')]

#print(test.shape)
X = df_orders_products[['user_id','product_id','order_id','add_to_cart_order','order_number','order_dow','order_hour_of_day']]

#X = df_orders_products.drop(['reordered'], axis=1)

#X = X.drop(['eval_set'], axis=1)

#X.isnull().any()
y = df_orders_products.reordered

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)
#xgb = XGBClassifier() 



#xgb.fit(X_train, y_train)  

#y_pred = xgb.predict(X_test)

#acc = accuracy_score(y_test,y_pred) 



#print(acc)
#classifiers = []



#model1 = XGBClassifier()

#classifiers.append(model1)

#model2 = SVC(gamma= "auto")

#classifiers.append(model2)

#model3 = DecisionTreeClassifier()

#classifiers.append(model3)

#model4  = RandomForestClassifier()

#classifiers.append(model4)

#print(classifiers)
random_grid = {'bootstrap': [True, False],

               'max_depth': [1,2,3,4,5, None],

               'max_features': ['auto', 'sqrt'],

               'min_samples_leaf': [1, 2, 4],

               'min_samples_split': [2, 5, 10],

               'n_estimators': [100,130, 150,180]}
#for clf in classifiers:

 #   clf.fit(X_train,y_train)

  #  y_pred = clf.predict(X_test)

   # acc = accuracy_score(y_test,y_pred) 

    #print("accuracy of %s is %s"% (clf,acc))"""

    
rf =RandomForestClassifier()



#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100,cv=2, verbose=2, random_state=42, n_jobs = -1)

#rf_random.fit(X_train,y_train)

#y_pred = rf_random.predict(X_test)

#acc = accuracy_score(y_test,y_pred)



#print(rf_random.best_params_)

#print(rf_random.best_score_)

#print(rf_random.best_estimator_)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10,cv=3, verbose=2, random_state=1, n_jobs = -1)

rf_random.fit(X_train,y_train)

y_pred = rf_random.predict(X_test)

acc = accuracy_score(y_test,y_pred)

print(acc)

print(rf_random.best_params_)

print(rf_random.best_score_)

print(rf_random.best_estimator_)
f1_score(y_test,y_pred)