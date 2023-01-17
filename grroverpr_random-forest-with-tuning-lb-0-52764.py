def set_rf_samples(n):

    """ Changes Scikit learn's random forests to give each tree a random sample of

    n random rows.

    """

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n))
def reset_rf_samples():

    """ Undoes the changes produced by set_rf_samples.

    """

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n_samples))
# Put these at the top of every notebook, to get automatic reloading and inline plotting

%reload_ext autoreload

%autoreload 2

%matplotlib inline
from xgboost import XGBRegressor
# This is required to make fastai library work on ec2-user 

# fastai library is not yet available using pip install

# pull it from github using below link

# https://github.com/fastai/courses

import warnings

warnings.filterwarnings('ignore')



#import sys

#sys.path.append("/Users/groverprince/Documents/msan/msan_ml/fastai/")

#sys.path.append("/home/groverprince/flogistix/fastai/")
# This file contains all the main external libs we'll use



import numpy as np

import pandas as pd



from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler

import operator

df_dtype = {}

df_dtype['train_id'] = 'int32'

df_dtype['item_condition_id'] = 'int32'

df_dtype['shipping'] = 'int8'

df_dtype['price'] = 'float32'

df_dtype
train = pd.read_csv("../input/train.tsv", delimiter='\t',

                   dtype= df_dtype)
train[:5]
train.apply(lambda x: x.nunique())
train.isnull().sum()
test = pd.read_csv("../input/test.tsv", delimiter='\t',

                   dtype= df_dtype)
test[:5]
train.describe()
test.describe()
len(test), len(train)
cat_vars = ['category_name', 'brand_name', 'shipping', 'name', 'item_description',

           'item_condition_id']
n = len(train)

n
train['is_train'] = 1

test['is_train'] = 0
train.rename(columns={'train_id':'id'}, inplace=True)

test.rename(columns={'test_id':'id'}, inplace=True)
train_test_combine = pd.concat([train.drop(['price'],axis =1),test],axis = 0)
for v in cat_vars: train_test_combine[v] = train_test_combine[v].astype('category').cat.as_ordered()
for v in cat_vars: train_test_combine[v] = train_test_combine[v].cat.codes
train_test_combine[:4]
df_test = train_test_combine.loc[train_test_combine['is_train']==0]

df_train = train_test_combine.loc[train_test_combine['is_train']==1]
df_test[:4]
df_train[:4]
df_train['price'] = train.price
df_train['price'] = df_train['price'].apply(lambda x: np.log(x) if x>0 else x)
df_train.drop(['id', 'is_train'],axis=1, inplace=True)
df_train[:3]
#df, y, nas, mapper = proc_df(df_train, 'price', do_scale=True)
#df[:4]
#y[:4]
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
x_train,y_train = df_train.drop(['price'],axis =1),df_train.price
#set_rf_samples(50000)
#m = RandomForestRegressor(n_jobs=-1, n_estimators=125)

#%time m.fit(x_train, y_train)
#preds = np.stack([t.predict(x_train) for t in m.estimators_])



#plt.plot([metrics.r2_score(y_train, np.mean(preds[:i+1], axis=0)) for i in range(125)]);

#plt.show()
grid = {

    'min_samples_leaf': [3,5,10,15,25,50,100],

    'max_features': ['sqrt', 'log2', 0.4, 0.5, 0.6]}

rf = RandomForestRegressor(n_jobs=-1, n_estimators=30,  random_state=42)
#gd = GridSearchCV(rf,grid, cv=3, verbose=50)
#gd.fit(x_train, y_train)
#gd.best_estimator_
#reset_rf_samples() 

rf2 = RandomForestRegressor(n_jobs=-1, n_estimators=100,  random_state=42, max_features=0.5, min_samples_leaf=3)
rf2.fit(x_train,y_train)
rf2.score(x_train,y_train)
df_test.drop(['is_train', 'id'], inplace=True, axis=1)
preds = rf2.predict(df_test)

preds = pd.Series(np.exp(preds))

submit = pd.concat([test.id,preds],axis=1)

submit.columns = ['test_id','price']
submit.to_csv('submit_rf_1.csv',index=False)
FileLink('submit_rf_1.csv')