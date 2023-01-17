import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
%matplotlib inline
import seaborn as sb

train = pd.read_table('../input/train.tsv')
test = pd.read_table('../input/test.tsv')
train = train.drop('train_id', 1)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import scipy

train["category_name"] = train["category_name"].fillna("Other")
train["brand_name"] = train["brand_name"].fillna("unknown")
test["category_name"] = test["category_name"].fillna("Other")
test["brand_name"] = test["brand_name"].fillna("unknown")


print("Category Encoders")
unique_categories = len(train['category_name'].unique())
count_category = CountVectorizer()
X_category = count_category.fit_transform(train["category_name"])
Test_unique_categories = len(test['category_name'].unique())
Test_count_category = CountVectorizer()
Test_X_category = Test_count_category.fit_transform(test["category_name"])
print("Brand encoders")
vect_brand = LabelBinarizer(sparse_output=True)
X_brand = vect_brand.fit_transform(train["brand_name"])
Test_vect_brand = LabelBinarizer(sparse_output=True)
Test_X_brand = Test_vect_brand.fit_transform(test["brand_name"])


print("Dummy Encoders")
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(train[[
    "item_condition_id", "shipping"]], sparse = True).values.astype('float32'))
Test_X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(test[[
    "item_condition_id", "shipping"]], sparse = True).values.astype('float32'))

X = scipy.sparse.hstack((X_dummies, 
                         X_brand,
                         X_category
                        )).tocsr()
T = scipy.sparse.hstack((Test_X_dummies, 
                         Test_X_brand,
                         Test_X_category
                        )).tocsr()

print ("Data preparation has been finished.")
train.head()
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, LogisticRegression
import lightgbm as lgb
import gc


params = {
    'learning_rate': 0.75,
    'application': 'regression',
    'max_depth': 3,
    'num_leaves': 100,
    'verbosity': -1,
    'metric': 'RMSE',
}


X_train = X
y_train = np.log1p(train["price"])
X_test = T
train_X, valid_X, train_y, valid_y = train_test_split(X_train, y_train, test_size = 0.1, random_state = 144) 

d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
watchlist = [d_train, d_valid]

model = lgb.train(params, train_set=d_train, num_boost_round=10000, valid_sets=watchlist, \
early_stopping_rounds=50, verbose_eval=100) 
preds = model.predict(X_test)

'''‘lsqr’ uses the dedicated regularized least-squares routine scipy.sparse.linalg.lsqr. 
It is the fastest but may not be available in old scipy versions. It also uses an iterative procedure'''
model = Ridge(solver = "lsqr", fit_intercept=False)

print("Fitting Model")
model.fit(X_train, y_train)
test["price"] = np.expm1(preds)
test[["test_id", "price"]].to_csv("submission.csv", index = False)
