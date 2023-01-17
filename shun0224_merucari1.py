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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
% matplotlib inline
types_dict_train = {'train_id':'int64', 'item_condition_id':'int8', 'price':'float64', 'shipping':'int8'}
types_dict_test = {'test_id':'int64', 'item_condition_id':'int8', 'shipping':'int8'}

train = pd.read_csv('../input/train.tsv', delimiter='\t', low_memory=True, dtype=types_dict_train)
test = pd.read_csv('../input/test.tsv', delimiter='\t', low_memory=True, dtype=types_dict_test)
target_col = 'price'
exclude_cols= ['price','train_id','test_id','name']
train_feature_cols= [col for col in train.columns if col not in exclude_cols]
# trainのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する
train.category_name = train.category_name.astype('category')
train.item_description = train.item_description.astype('category')
train.name = train.name.astype('category')
train.brand_name = train.brand_name.astype('category')
 
# testのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する
test.category_name = test.category_name.astype('category')
test.item_description = test.item_description.astype('category')
test.name = test.name.astype('category')
test.brand_name = test.brand_name.astype('category')
train.dtypes, test.dtypes
train.name = train.name.cat.codes
train.category_name = train.category_name.cat.codes
train.brand_name = train.brand_name.cat.codes
train.item_description = train.item_description.cat.codes
test.name = test.name.cat.codes
test.category_name = test.category_name.cat.codes
test.brand_name = test.brand_name.cat.codes
test.item_description = test.item_description.cat.codes
train['price'] = train['price'].apply(lambda x: np.log(x) if x>0 else x)

y = np.array(train[target_col])
X = np.array(train[train_feature_cols])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
def get_gfs_feature_indices(X, y,features, clf):
    X_train_, X_test_, y_train_,  y_test_ = train_test_split(X, y,test_size=0.3,random_state=1234)
    feature_indices = {feature: idx for idx, feature in enumerate(features)}
    features = set(features)
    last_mse = np.inf
    chosen_features = set()
    while len(chosen_features) < len(features):
        mse_features = []
        for feature in (features - chosen_features):
            candidates = chosen_features.union(set([feature]))
            indices = [feature_indices[feature] for feature in candidates]
            clf.fit(X_train_[:, indices], y_train_)
            y_pred = clf.predict(X_test_[:, indices])
            mse = mean_squared_error(y_test_, y_pred)
            mse_features += [(mse,feature)]
        mse, feature = min(mse_features)
        if mse >= last_mse:
            break
        last_mse = mse
        print('Newly Added Feature: {},\t MSE Score: {}'.format(feature, mse))
        chosen_features.add(feature)
    return [feature_indices[feature] for feature in chosen_features]
selected_feature_index = get_gfs_feature_indices(X = X_train,y = y_train,features = train_feature_cols, clf= RandomForestRegressor())
rf = RandomForestRegressor()
params = { 'max_depth':[5,10,20], 'min_samples_leaf':[100,200]}
gscv = GridSearchCV(rf, param_grid=params, cv=3, scoring='neg_mean_squared_error', n_jobs =-1)
gscv.fit(X_train[:, selected_feature_index], y_train)
test_feature_cols = ["item_condition_id","category_name","brand_name","shipping"]
X = np.array(test[test_feature_cols])
preds = gscv.predict(X)
np.exp(preds)
preds = pd.Series(np.exp(preds))
submit = pd.concat([test.test_id, preds], axis=1)
submit.columns = ['test_id', 'price']
submit.to_csv('submit_rf_base1.csv', index=False)
