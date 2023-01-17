import pandas as pd

import numpy as np

import lightgbm as lgb

import matplotlib.pyplot as plt
sales = pd.read_csv('../input/retaildataset/sales data-set.csv')

stores = pd.read_csv('../input/retaildataset/stores data-set.csv')

features = pd.read_csv('../input/retaildataset/Features data set.csv')
sales.head()
stores.head()
features.head()
features.info()
df = pd.merge(sales,stores,on=['Store'],how='left')
df = pd.merge(df,features,on=['Store','Date'],how='left')
df.fillna(0,inplace=True)
def mon(x):

    return int(x.split('/')[1])

df['month'] = df['Date'].apply(mon) 
df.info()
df.head()
df.Type.value_counts(dropna=False)
df.Type = df.Type.replace({'A':0,'B':1,'C':2})
df.isna().any().sum()
#importing libraries

from sklearn.ensemble import RandomForestRegressor

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import tree

import graphviz

import random 

random.seed(3)
X_train = df.drop(['Weekly_Sales','Date'],axis = 1)

y_train = df['Weekly_Sales']



X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,random_state = 3,test_size = 0.2)

r_randomForest = RandomForestRegressor(random_state=0, max_depth=5, min_samples_split=5).fit(X_train,y_train)
print(mean_squared_error(y_test,r_randomForest.predict(X_test)))
#permutation importance

import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(r_randomForest, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
#preparing data for lightgbm

lgb_train = lgb.Dataset(X_train,y_train)

lgb_test = lgb.Dataset(X_test,y_test)
#Let's set parameters for our model

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'rmse'},

    'num_leaves': 31,

    'learning_rate': 0.01,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

}

#training our lightgbm model

model = lgb.train(params,lgb_train,num_boost_round=10000,valid_sets=lgb_test,early_stopping_rounds=100)
#model.feature_importance()

feature = pd.DataFrame({'features':X_train.columns,'importance':model.feature_importance()})
X_train
plt.barh(feature['features'],feature['importance'])

plt.show()