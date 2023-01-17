## Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier, Pool, cv

#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.linear_model import LogisticRegression
#import lightgbm as lgb
# Load CSV

data = pd.read_csv('/kaggle/input/hackerearth-pet-adoption-dataset/train.csv')
test = pd.read_csv('/kaggle/input/hackerearth-pet-adoption-hackathon-dataset/test.csv')

origial=data.copy()

print('Train Data ---')
display(data.head(3))
print('Test Data ---')
display(data.tail(3))
## combine train and test data

data_com = pd.concat((data, test))
## checking for null values

data_com.isnull().sum()
## taking a look at some information about dataset

data_com.info()
## modify listing and issue date coloumn from object to datetime format

data_com.issue_date = pd.to_datetime(data_com.issue_date)
data_com.listing_date = pd.to_datetime(data_com.listing_date)

## creating a new coloumn (time duration for delivery)

data_com['time_diff'] = data_com.listing_date - data_com.issue_date
data_com['time_diff'] = data_com.time_diff.dt.total_seconds()

## applying log

data_com['time_diff'] = np.log1p(data_com['time_diff'])
col = ['color_type']
for c in col:
    le = LabelEncoder()
    data_com[c]=le.fit_transform(data_com[c])
    
## coverting height from cm to m

data_com['height'] = data_com['height(cm)']/100
data_com.drop(['height(cm)'], axis=1, inplace=True)
data_com.head(3)
## I found pet id to be useless as all the values are unique but some people 
## used it after splitting it ( ANSL_6 and 9903)

data_ = data_com.drop(['pet_id'], axis=1)
## Some new features which you can try 

#data_['L/H'] = np.round(data_['length(m)'] / data_['height'], 3)
#data_['X1/X2'] = data_['X1'] / data_['X2']
#data_ = data_com.drop(['pet_id'], axis=1)
#data_['1'] = np.round(data_['length(m)'] * data_['height'], 3)
#data_['2'] = np.round(data_['length(m)'] + data_['height'], 3)
#data_['3'] = np.round(data_['length(m)'] - data_['height'], 3)
#data_['4'] = data_['X1'] * data_['X2']
#data_['5'] = data_['X1'] + data_['X2']
#data_['6'] = data_['X1'] - data_['X2']
from xgboost import XGBClassifier
data_condition_null = data_[data_.condition.isnull()==True]
data_condition_not_null = data_[data_.condition.isnull()==False]
x = data_condition_not_null[['X1', 'X2', 'color_type', 'time_diff', 'length(m)', 'height']]
y = data_condition_not_null['condition']
to_predict = data_condition_null[['X1', 'X2', 'color_type', 'time_diff', 'length(m)', 'height']]
clf_xg_fill = XGBClassifier()
clf_xg_fill.fit(x,y)
abc = clf_xg_fill.predict(to_predict)
sns.countplot(abc)
## Splitting Data into train and test and fiiling missing values

train = data_[data_.pet_category.isnull()==False]
train = train.fillna(3)

test_ = data_[data_.pet_category.isnull()==True]
test_.drop(['breed_category', 'pet_category'], axis=1,inplace=True)

x = train[['X1', 'X2', 'color_type', 'condition', 'time_diff', 'length(m)', 'height']]
y1 = train['pet_category']
y2 = train['breed_category']
x['condition'] = x['condition'].astype('int')
## grid search for XGBClassifier

params = {'min_child_weight':[1,5,10],
         'gamma':[0.5,1, 2, 5],
         'max_depth':[3,5],
         'subsample':[0.6, 1.0],
         'learning_rate':[0.001,0.01,0.05,0.1],
         'n_estimators':[100,300,500,700,900,1000]}
xgb = XGBClassifier()
random_search = RandomizedSearchCV(xgb, param_distributions=params, n_jobs=-1, cv=3, verbose=5)
r_s = random_search.fit(x,y2)
r_s.best_estimator_
## grid search for RandomForestClassifier

params = {'bootstrap':[True, False], 
         'max_depth':[10,30,50,70,90,100,None],
         'max_features':['auto', 'sqrt'],
         'min_samples_leaf':[1,2,4],
         'min_samples_split':[2,5,10],
         'n_estimators':[200,600,1000,1400,1800]}
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(rf, param_distributions=params, n_jobs=-1, verbose=3, cv=3)
r_f = rf_random.fit(x,y2)

r_f.best_estimator_
## grid search for CatBoostClassifier

cat_features = ['X2', 'color_type', 'condition']
grid = {
    'learning_rate': [0.05, 0.07, 0.09, 0.3],
    'depth': [5, 6, 7],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
}
train_pool = Pool(x, label=y2, cat_features=cat_features)
model = CatBoostClassifier(
        early_stopping_rounds=100,
        has_time=True,
        iterations=5000
    )

model.randomized_search(grid, X=train_pool)

cat_features = ['condition', 'color_type', 'X1']
params = {'depth': 7,
          'l2_leaf_reg': 3,
          'learning_rate': 0.07,
          'grow_policy': 'SymmetricTree',
         'cat_features': cat_features,
         'verbose': 200,
         'eval_metric': 'Accuracy'}
xtrain, xtest, ytrain, ytest = train_test_split(x, y2, random_state=12)

clf_rf = RandomForestClassifier(min_samples_leaf=2, n_estimators=200)
clf_cat = CatBoostClassifier(**params)
clf_xg =XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=5, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=3,
              min_child_weight=5, missing=np.nan, monotone_constraints='()',
              n_estimators=300, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=0.6,
              tree_method='exact', validate_parameters=1, verbosity=None)

#clf_lgb = lgb.LGBMClassifier()
#clf_ad = AdaBoostClassifier()

clf2 = VotingClassifier(estimators=[('rf', clf_rf), ('xgb', clf_xg),('cat', clf_cat)], voting='soft')

clf2.fit(x,y2)
## Adding bread_category to train_data

x['breed_category'] = clf2.predict(x)
x['breed_category'] = x['breed_category'].astype('int')
## grid search for RandomforestClassifier

params = {'bootstrap':[True, False], 
         'max_depth':[10,30,50,70,90,100,None],
         'max_features':['auto', 'sqrt'],
         'min_samples_leaf':[1,2,4],
         'min_samples_split':[2,5,10],
         'n_estimators':[200,600,1000,1400,1800]}
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(rf, param_distributions=params, n_jobs=-1, verbose=3, cv=3)
r_f = rf_random.fit(x,y1)
r_f.best_estimator_
## grid search for XGBClassifier

params = {'min_child_weight':[1,5,10],
         'gamma':[0.5,1, 2, 5],
         'max_depth':[3,5],
         'subsample':[0.6, 1.0],
         'learning_rate':[0.001,0.01,0.05,0.1],
         'n_estimators':[100,300,500,700,900,1000]}
xgb = XGBClassifier()
random_search = RandomizedSearchCV(xgb, param_distributions=params, n_jobs=-1, cv=3, verbose=5)
r_s = random_search.fit(x,y1)
r_s.best_estimator_
## grid search for CatBoostClassifier

cat_features = ['X2', 'color_type', 'condition', 'breed_category']
grid = {
    'learning_rate': [0.05, 0.07, 0.09, 0.3],
    'depth': [5, 6, 7],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
}
train_pool = Pool(x, label=y1, cat_features=cat_features)
model = CatBoostClassifier(
        early_stopping_rounds=100,
        has_time=True,
        iterations=5000
    )

model.randomized_search(grid, X=train_pool)
cat_features = ['condition', 'color_type', 'breed_category', "X1"]
params = {'depth': 6,
         'l2_leaf_reg': 9,
         'learning_rate': 0.07,
         'grow_policy': 'Depthwise',
         'cat_features': cat_features,
         'verbose':200,
         'eval_metric':'Accuracy'}

xtrain, xtest, ytrain, ytest = train_test_split(x, y1, random_state=12)

clf_rf = RandomForestClassifier(max_depth=90, min_samples_split=10, n_estimators=1800)

clf_cat = CatBoostClassifier(**params)

clf_xg = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=1, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=10, missing=np.nan, monotone_constraints='()',
              n_estimators=300, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1.0,
              tree_method='exact', validate_parameters=1, verbosity=None)

#clf_ad = AdaBoostClassifier()

#clf_lgb = lgb.LGBMClassifier()

clf1 = VotingClassifier(estimators=[('rf', clf_rf), ('xgb', clf_xg), ('cat', clf_cat)], voting='soft')

clf1.fit(x,y1)
test_ = test_.drop(['listing_date', 'issue_date'], axis=1)
test_ = test_.fillna(4)

test_[['X1', 'X2', 'color_type', 'condition', 'time_diff','length(m)',
       'height']] = test_[['X1', 'X2', 'color_type', 'condition', 'length(m)', 'time_diff',
       'height']]
test_.columns = ['X1', 'X2', 'color_type', 'condition', 'time_diff','length(m)',
       'height']
test_['condition'] = test_['condition'].astype('int')
test_['color_type'] = test_['color_type'].astype('int')
test_['X1'] = test_['X1'].astype('int')
y2 = clf2.predict(test_)
y2 = np.ravel(y2)

test_['breed_category'] = y2
test_['breed_category'] = test_['breed_category'].astype('int')
y1 = clf1.predict(test_)
y1 = np.ravel(y1)
dataframe = pd.DataFrame({'pet_id':test.pet_id, 
                         'breed_category':y2,
                         'pet_category':y1})
dataframe.breed_category = dataframe.breed_category.astype('int')
dataframe.pet_category = dataframe.pet_category.astype('int')

#dataframe.to_csv('submission/votingclassifier21.csv', index=False)