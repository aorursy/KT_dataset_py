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
!apt-get install p7zip
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z
!p7zip -d -f -k /kaggle/input/mercari-price-suggestion-challenge/test.tsv.7z
def rmsle(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))
import os,sys
import pandas as pd
df_train = pd.read_csv('train.tsv', sep='\t')
def SplitCategories(catstr):
    """
    There are many compound categories, i.e. "décor & storage".
    This is a logical OR between "decor" and "storage"
    We can split these into two categories and put 1 in both columns
    """
    # lowercase
    catstr = catstr.lower()
    # split on /&,
    wl =  wl = re.split('/|&|,',catstr)
    # strip leading/trailing spaces and trailing s
    # (poor man's singularization)
    return [k.strip().rstrip('s') for k in wl]
# Split the string like 'Men/Athletic Apparel/Jackets'
# into a (lowercase) list 'men, athletic apparel, jackets'
import re
df_train['category_list'] = df_train['category_name'].astype(str).apply(SplitCategories)
# ['category_name']
import scipy
from sklearn.preprocessing import MultiLabelBinarizer
lb_category = MultiLabelBinarizer(sparse_output=True)
csr_category_onehot = lb_category.fit_transform(df_train['category_list'].fillna('nan'))
#scipy.sparse.save_npz("csr_category.npz",csr_category_onehot)
# ['brand_name']
from sklearn.preprocessing import LabelBinarizer
lb_brand = LabelBinarizer(sparse_output=True)
csr_brand = lb_brand.fit_transform(df_train['brand_name'].fillna('nan'))
#scipy.sparse.save_npz('csr_brand.npz', csr_brand)
# ['item_condition_id', 'shipping']
df_train_condship = pd.get_dummies(df_train[['item_condition_id', 'shipping']], columns  = ['item_condition_id', 'shipping'], sparse=True)
scr_condship = scipy.sparse.csr_matrix(df_train_condship.values)
# ['item_description']
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features=40000, ngram_range=(1, 3), stop_words='english')
scr_description = tv.fit_transform(df_train['item_description'].astype(str).apply(lambda x: x.replace('No description yet','')))
['category_name','brand_name','item_condition_id', 'shipping']
scr_combined = scipy.sparse.hstack([csr_category_onehot,csr_brand,scr_condship,scr_description])
y = df_train['price'].values
y_log = np.log(1+y)
# Create lgb dataset with target values log(1+y)
import lightgbm as lgb
# d_train = lgb.Dataset(scr_combined.astype(float), 
#                       label=y_log,
#                       free_raw_data=False)
d_train = lgb.Dataset(scr_combined, 
                      label=y_log,
                      free_raw_data=False)
# param_sets =     [  { 'num_leaves':     [2**3],
#                     'objective' :    'regression',},  
                  
#                     { 'num_leaves':     [2**4],
#                         'objective' :    'regression',},
                    
#                     { 'num_leaves':     [2**4],
#                         'max_depth' :     [4],
#                         'objective' :    'regression',},
# ]
# # GridSearchCV does not support sparse matrices
# # hence implement manual selection of the best estimator
# best_error = 100
# n_estimators = 1000
# for k,params in enumerate(param_sets):
#     bst = lgb.train(params, d_train, n_estimators)
#     # compute prediction [on TRAIN data] in logarithmic domain 
#     y_log_pred = bst.predict(scr_combined.astype(float))
#     # convert predictions [on TRAIN data] to linear domain
#     y_pred = np.exp(y_log_pred) - 1
#     # compute RMSLE error
#     err = rmsle(y, y_pred)
#     if err < best_error:
#         best_error = err
#         best_estimator = bst
#     print(f"Fininshed evaluating {k}-th set of parameters")

# print(f"Best error: {best_error}")
# Fininshed evaluating 0-th set of parameters
# Fininshed evaluating 1-th set of parameters
# Fininshed evaluating 2-th set of parameters
# Best error: 0.48675887913909316
#
# > bst.params
# {'num_leaves': [16], 'max_depth': [4], 'objective': 'regression'}
params =     {
        'num_leaves':     [2**4],
        'max_depth' :     [4],
        'objective' :    'regression',
        'device'    :    'cpu'
}
import lightgbm as lgb
n_estimators = 1000
bst = lgb.train(params, d_train, n_estimators)
# compute prediction [on TRAIN data] in logarithmic domain 
y_log_pred = bst.predict(scr_combined.astype(float))
# convert predictions [on TRAIN data] to linear domain
y_pred = np.exp(y_log_pred) - 1
# compute RMSLE error
err = rmsle(y, y_pred)
err

#import joblib
#joblib.dump(bst,'lgbm.model.pkl')
## load model
## bst = joblib.load('lgbm.model.pkl')
!unzip -f /kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip
#release some memory
del df_train, scr_combined, csr_category_onehot, csr_brand, scr_condship, scr_description
df_test = pd.read_csv('test_stg2.tsv', sep='\t')
df_test.shape
# ['category_name']
df_test['category_list'] = df_test['category_name'].astype(str).apply(SplitCategories)
test_category_onehot = lb_category.transform(df_test['category_list'].fillna('nan'))
# ['brand_name']
test_brand = lb_brand.transform(df_test['brand_name'].fillna('nan'))
# ['item_condition_id', 'shipping']
df_test_condship = pd.get_dummies(df_test[['item_condition_id', 'shipping']], columns  = ['item_condition_id', 'shipping'], sparse=True)
test_condship = scipy.sparse.csr_matrix(df_test_condship.values)
test_description = tv.transform(df_test['item_description'].astype(str).apply(lambda x: x.replace('No description yet','')))
test_combined = scipy.sparse.hstack([test_category_onehot,test_brand,test_condship,test_description])
y_log_pred = bst.predict(test_combined)
y_pred = np.exp(y_log_pred) - 1
test_result_df = pd.DataFrame(data={'test_id':df_test.index,'price':y_pred})
# create submission
test_result_df.to_csv("submission.csv", index = False)
