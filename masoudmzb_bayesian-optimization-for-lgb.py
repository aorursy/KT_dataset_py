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



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns; sns.set()

from sklearn.externals import joblib



from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge



from xgboost import XGBRegressor



import gc

from itertools import product

import time



from bayes_opt import BayesianOptimization

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

from bayes_opt.logger import JSONLogger

from bayes_opt.event import Events
all_data = pd.read_pickle("/kaggle/input/eda-with-feature-engineering/all_data.pkl")

all_data.head()
test_items = all_data.loc[all_data['date_block_num']==34,'item_id'].unique()

train_items = all_data.loc[all_data['date_block_num']<34,'item_id'].unique()

items_in_test_and_not_in_train = set(test_items).difference(set(train_items))

print('Items in test and not in train: {0}'.format(len(items_in_test_and_not_in_train)))

items_in_train_and_not_in_test = set(train_items).difference(set(test_items))

print('Items in train and not in test: {0}'.format(len(items_in_train_and_not_in_test)))



test_shops = all_data.loc[all_data['date_block_num']==34,'shop_id'].unique()

print('Number of unique shops: {0}'.format(len(test_shops)))

missing_shop_item_count = 378 * 42 # all missing item per shop ===> 15876

index_cols = ['shop_id', 'item_id', 'date_block_num']



grid = []

for block_num in all_data.loc[all_data['date_block_num']<34, 'date_block_num'].unique():

    print(block_num)

    

    zero_target_df = all_data[(all_data['date_block_num'] == block_num) & (all_data['target']==0) & 

                              (all_data['item_id'].isin(items_in_train_and_not_in_test))]

    

    idx_to_delete = zero_target_df.sample(missing_shop_item_count, random_state=block_num).index

    all_data.drop(idx_to_delete, inplace=True)

    temp = np.array(list(product(*[test_shops, items_in_test_and_not_in_train, [block_num]])),dtype='int32')

    grid.append(temp)

    

    del zero_target_df

    del idx_to_delete

    del temp

    gc.collect()



#     I think grid is all items that a specific shop(in train data) didn't have in each month

# یعنی اجناسی که هر مغازه در ماه های قبل ۳۴ نفروخته (یا همون نداشته که بفروشه )

#  non of grid rows are in all_data

grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

    

    
grid['shop_id'] = grid['shop_id'].astype(np.int16)

grid['item_id'] = grid['item_id'].astype(np.int32)

grid['date_block_num'] = grid['date_block_num'].astype(np.int8)
all_data = pd.concat([all_data, grid], ignore_index=True, sort=False, keys=index_cols)

all_data[['item_shop_last_sale', 'item_last_sale']].fillna(-1, inplace=True) #-1 is default value in this columns

all_data.fillna(0, inplace=True)



del grid

del test_items

del test_shops

del train_items

del items_in_test_and_not_in_train

del items_in_train_and_not_in_test

gc.collect()
all_data
all_data['is_december'] = all_data['is_december'].astype(np.int8)

all_data['item_category_id'] = all_data['item_category_id'].astype(np.int8)

all_data['type_code'] = all_data['type_code'].astype(np.int8)

all_data['subtype_code'] = all_data['subtype_code'].astype(np.int8)

all_data['city_code'] = all_data['city_code'].astype(np.int16)



all_data['month'] = all_data['month'].astype(np.int8)

all_data['days'] = all_data['days'].astype(np.int8)

all_data['item_shop_last_sale'] = all_data['item_shop_last_sale'].astype(np.int8)

all_data['item_last_sale'] = all_data['item_last_sale'].astype(np.int8)

all_data['item_shop_first_sale'] = all_data['item_shop_first_sale'].astype(np.int8)

all_data['item_first_sale'] = all_data['item_first_sale'].astype(np.int8)
# we added some rows to balance ditribution of our dataset so we need put those new rows in right position

# put added rows in right position

all_test_data = all_data[all_data['date_block_num'] == 34]

all_data = all_data[all_data['date_block_num'] < 34]

all_data.sort_values(['date_block_num'], inplace=True)

all_data = pd.concat([all_data, all_test_data], ignore_index=True, sort=False, keys=index_cols)



del all_test_data

gc.collect()

dates = all_data['date_block_num']



last_block = dates.max()

print('Test `date_block_num` is {0}'.format(last_block))



X_train = all_data.loc[dates <  last_block]

X_test =  all_data.loc[dates == last_block]



y_train = all_data.loc[dates <  last_block, 'target'].values

y_test =  all_data.loc[dates == last_block, 'target'].values



X_valid_train = all_data.loc[dates <  last_block-1]

X_valid_test =  all_data.loc[dates == last_block-1]



y_valid_train = all_data.loc[dates <  last_block-1, 'target'].values

y_valid_test =  all_data.loc[dates == last_block-1, 'target'].values



all_data.to_pickle('all_data.pkl') # will use it later. Now free RAM



del dates

del all_data

gc.collect()
# delete some rows from test

columns_to_delete = ['date_block_num', 'target']

X_valid_train = X_valid_train.drop(columns_to_delete, axis=1)

X_valid_test = X_valid_test.drop(columns_to_delete, axis=1)



X_train = X_train.drop(columns_to_delete, axis=1)

X_test = X_test.drop(columns_to_delete, axis=1)
#  we need list of column_names in lgb.Dataset()

predictors = X_valid_train.columns.tolist()
bayesian_tr_index, bayesian_val_index = list(StratifiedKFold(2, random_state=12, shuffle=True).split(X_valid_train, y_valid_train))[0]
# in bayesian optimization we need to have a black box. this black box is our algorithm which we want to optimize

def lgb_black_box(

    num_leaves,  # int

    min_data_in_leaf,  # int

    learning_rate,

    min_sum_hessian_in_leaf,    # int  

    feature_fraction,

    lambda_l1,

    lambda_l2,

    min_gain_to_split,

    max_depth):

    

    # lgb need some inputs as int but BayesianOptimization library send continuous values values. so we change type.



    num_leaves = int(num_leaves)

    min_data_in_leaf = int(min_data_in_leaf)

    max_depth = int(max_depth)

    

    # all this hyperparameter values are just for test. our goal in this kernel is how to use bayesian optimization

    # you can see lgb documentation for more info about hyperparameters

    params = {

        'num_leaves': num_leaves,

        'max_bin': 63,

        'min_data_in_leaf': min_data_in_leaf,

        'learning_rate': learning_rate,

        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,

        'bagging_fraction': 1.0,

        'bagging_freq': 5,

        'feature_fraction': feature_fraction,

        'lambda_l1': lambda_l1,

        'lambda_l2': lambda_l2,

        'min_gain_to_split': min_gain_to_split,

        'max_depth': max_depth,

        'save_binary': True, 

        'seed': 1337,

        'feature_fraction_seed': 1337,

        'bagging_seed': 1337,

        'drop_seed': 1337,

        'data_random_seed': 1337,

        'objective': 'regression',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'rmse',

        'is_unbalance': True,

        'boost_from_average': False, 

    }

    

    train_data = lgb.Dataset(X_valid_train.iloc[bayesian_tr_index].values,

                            label = y_valid_train[bayesian_tr_index],

                            feature_name=predictors,

                            free_raw_data = False)

    

    

    validation_data = lgb.Dataset(X_valid_train.iloc[bayesian_val_index].values,

                                 label= y_valid_train[bayesian_val_index],

                                 feature_name=predictors,

                                 free_raw_data=False)

    

    num_round = 5000

    clf = lgb.train(params, train_data, num_round, valid_sets = [validation_data], verbose_eval=250,

                 early_stopping_rounds = 50)

    

    predictions = clf.predict(X_valid_train.iloc[bayesian_val_index].values,

                              num_iteration = clf.best_iteration)

    

#      we need to compute a regression score. roc_auc_score is a classification score. we can't use it

#     score = metrics.roc_auc_score(y_valid_train[bayesian_val_index], predictions)

    mse = mean_squared_error(y_valid_train[bayesian_val_index], predictions)

    rmse = np.sqrt(mse)

#     our bayesian optimization expect us to give him increasing number to understand this is getting better

    return -rmse
# these ranges are not best range for this competition, I just use these base ranges

LGB_bound = {

    "num_leaves" : (5, 20),

    "min_data_in_leaf" : (5, 20),

    "learning_rate" : (0.01, 0.3),

    "min_sum_hessian_in_leaf" : (0.00001, 0.01),

    "feature_fraction" : (0.05, 0.5),

    "lambda_l1" : (0, 5.0),

    "lambda_l2" : (0, 5.0),

    'min_gain_to_split': (0, 1.0),

    'max_depth':(3,15)

}
from bayes_opt import BayesianOptimization



#  we have 3 parameters for this object. first is function. second is ranges. third is random_state (no matter)

optimizer = BayesianOptimization(

    f=lgb_black_box,

    pbounds = LGB_bound,

    random_state = 13

)

print(optimizer.space.keys)
init_points = 3

n_iter = 3



optimizer.maximize(init_points = init_points, n_iter = n_iter)
optimizer.max["params"]
# here i say hey optimizer! search for this new parameter to see if they are really better or not.

# probe = کاوش

#  tmp code

#  feature fraction = 0.3064, l1=  2.659,  l2 =   0.3892, learning =  0.1054,

# max_depth = 14.76, min_da  19.7,   min_ga = 0.6548,   min_su = 0.000626, num_lea 19.06



optimizer.probe(

    params = {

        'feature_fraction': 0.3064, 

            'lambda_l1': 2.659, 

            'lambda_l2': 0.3892, 

            'learning_rate': 0.1054, 

            'max_depth': 14.76, 

            'min_data_in_leaf': 19.7, 

            'min_gain_to_split': 0.6548, 

            'min_sum_hessian_in_leaf': 0.000626, 

            'num_leaves': 19.06

    },

    lazy = False

)

# if lazy= True  it will run next time I say .maximize

# if lazy = False it will run the optimizing right now.
optimizer.max["params"]


optimized_lgb_params = {

        'num_leaves': int(optimizer.max["params"]["num_leaves"]),

        'max_bin': 63,

        'min_data_in_leaf': int(optimizer.max["params"]["min_data_in_leaf"]),

        'learning_rate': optimizer.max["params"]["learning_rate"],

        'min_sum_hessian_in_leaf': optimizer.max["params"]["min_sum_hessian_in_leaf"],

        'bagging_fraction': 1.0,

        'bagging_freq': 5,

        'feature_fraction': optimizer.max["params"]["feature_fraction"],

        'lambda_l1': optimizer.max["params"]["lambda_l1"],

        'lambda_l2': optimizer.max["params"]["lambda_l2"],

        'min_gain_to_split': optimizer.max["params"]["min_gain_to_split"],

        'max_depth': int(optimizer.max["params"]["max_depth"]),

        'save_binary': True, 

        'seed': 1337,

        'feature_fraction_seed': 1337,

        'bagging_seed': 1337,

        'drop_seed': 1337,

        'data_random_seed': 1337,

        'objective': 'regression',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'rmse',

        'is_unbalance': True,

        'boost_from_average': False, 

    }

    
nfold = 5

import gc

gc.collect()

skf2 = StratifiedKFold(n_splits = nfold, shuffle = True, random_state=68)
predictions1 = np.zeros((len(y_test), nfold))

i = 1

for train_index, val_index in skf2.split(X_train, y_train):

    train_set_lgb = lgb.Dataset(X_train.iloc[train_index][predictors].values,

                                label= y_train[train_index],

                                feature_name= predictors,

                                free_raw_data=False)

    

    val_set_lgb = lgb.Dataset(X_train.iloc[val_index][predictors].values,

                                label= y_train[val_index],

                                feature_name= predictors,

                                free_raw_data=False)

    clf = lgb.train(optimized_lgb_params, train_set_lgb, 5000, valid_sets = [val_set_lgb],

                   verbose_eval=250, early_stopping_rounds = 50)

    

    predictions1[:,i-1] += clf.predict(X_test[predictors], num_iteration=clf.best_iteration)

    i = i + 1





su=[sum(i) for i in predictions1]

newList = [ x / 5 for x in su]

newList
clipedList = [20 if x > 20 else x  for x in newList ]
submit3 = pd.DataFrame({'ID':range(214200), 'item_cnt_month': clipedList})

submit3.to_csv('submit3.csv', index=False)

train_index2, val_index2 = list(StratifiedKFold(2, random_state=12, shuffle=True).split(X_train, y_train))[0]
# prediction with 1 time of prediction.

predictions = np.zeros((len(y_test), nfold))





train_set_lgb = lgb.Dataset(X_train.iloc[train_index2][predictors].values,

                            label= y_train[train_index2],

                            feature_name= predictors,

                            free_raw_data=False)

    

val_set_lgb = lgb.Dataset(X_train.iloc[val_index2][predictors].values,

                            label= y_train[val_index2],

                            feature_name= predictors,

                            free_raw_data=False)

clf = lgb.train(optimized_lgb_params, train_set_lgb, 5000, valid_sets = [val_set_lgb],

                verbose_eval=250, early_stopping_rounds = 50)

    

predictions = clf.predict(X_test[predictors], num_iteration=clf.best_iteration)
final = predictions.clip(0,20)
predictions



submit = pd.DataFrame({'ID':range(len(predictions)), 'item_cnt_month': final})

submit.to_csv('submit.csv', index=False)
submit