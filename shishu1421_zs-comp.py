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
import pandas as pd  

data = pd.read_csv('../input/cristiano7/data.csv')

data.head()
data['shot_id_number'] = range(1,30698)
data = data[['location_x','location_y','power_of_shot','knockout_match','distance_of_shot','is_goal','area_of_shot','shot_basics','range_of_shot'

            ,'home/away','lat/lng','type_of_shot','type_of_combined_shot','shot_id_number','distance_of_shot.1']]
def home_away(string):

    if '@' in str(string):

        return 'away'

    else:

        return 'home'
data['home_away'] = data['home/away'].apply(home_away)
data.drop(['home/away','lat/lng'],axis=1,inplace=True)

data.head()
import math

data['distance'] = data['location_x']**2 + data['location_y']**2

data['distance'] = data.distance.apply(math.sqrt)
data.drop(['location_x','location_y'],axis=1,inplace=True)

data.head()
data = pd.get_dummies(data)

train = data[data['is_goal'].notnull()]

#sub = data[data['is_goal'].isna() & data['shot_id_number'].notnull()]
sub = pd.read_csv('../input/zsdata/sample_submission.csv')

sub = sub.drop(['is_goal'],axis = 1)
result = pd.merge(sub, data, how='inner', on=['shot_id_number'])

result.head()





from sklearn.model_selection import StratifiedKFold

kfold = 5

skf = StratifiedKFold(n_splits=kfold, random_state=42)
'''params = {

    'min_child_weight': 6.0,

    'objective': 'binary:logistic',

    'max_depth': 11,

    'colsample_bytree': 0.75,

    'subsample': 0.8,

    'eta': 0.05,

    'gamma': 5,

    'eval_metric' : 'mae',

    'silent': 1,

    'num_boost_round' : 700

    }'''

params = {

    'booster':'gbtree',

           'subsample': 1.0,

          'objective': 'binary:logistic',

          'min_child_weight': 5, 

          'eta': 0.01,

          'max_depth': 5,

          'gamma': 5,

          'colsample_bytree': 0.6,

          'eval_metric' : 'mae'

         }
X =  pd.get_dummies(train.drop(["is_goal","shot_id_number"],axis = 1)).values

y = train.is_goal.values
#def error()

from sklearn.metrics import mean_absolute_error

def evalerror(preds, dtrain):

    labels = dtrain.get_label()

    return 'my-error', (1/(1+mean_absolute_error(labels,preds)))
result1 = pd.DataFrame()

#sub_data = pd.read_csv('../input/data.csv')

#sub_data = sub_data[sub_data['is_goal'].isna()]

result1["shot_id_number"] = result.shot_id_number

result1["is_goal"] = 0

#test.drop(["unique_id","cns_score_description"],axis=1,inplace=True)
import xgboost as xgb

from sklearn.metrics import roc_auc_score

for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    print('[Fold %d/%d]' % (i + 1, kfold))

    X_train, X_valid = X[train_index], X[test_index]

    y_train, y_valid = y[train_index], y[test_index]

    # Convert our data into XGBoost format

    d_train = xgb.DMatrix(X_train, y_train)

    d_valid = xgb.DMatrix(X_valid, y_valid)

    d_test = xgb.DMatrix(result.drop(['is_goal','shot_id_number'],axis=1).values)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    mdl = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=1000, maximize=True,verbose_eval = 100,feval=evalerror)

    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))

    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)

    result1['is_goal'] += p_test/kfold
result1.head()



result1.to_csv('result_zs1.csv',index=False)