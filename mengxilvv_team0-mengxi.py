import pandas as pd

import os

#os.getcwd()
ks_train = pd.read_csv('../input/2020caohackathon/train_set.csv')

ks_train['train']=1

#ks_train['outcome']

#ks_train.shape
def creator_info_help(data):

    creator_info = data.groupby('creator_id').agg(creator_freq = ('id','count'))

    return creator_info
ks_test = pd.read_csv('../input/2020caohackathon/test_set.csv')

ks_test['train']= 0

#ks_test.shape
ks = ks_train.append(ks_test, ignore_index=True)

ks = ks.merge(creator_info_help(ks),left_on='creator_id',right_on='creator_id',how='left').fillna(0)
ks['name_len']=[len(x) for x in ks.creator_name]
cat_features = ['main_category','sub_category', 'currency', 'country','disable_communication','location_state',

                'staff_pick','creator_register','show_feature_image','currency_trailing_code',

                'category-1','category-2', 'category-3']

ks[cat_features].head(1)
from sklearn.preprocessing import LabelEncoder



#cat_features = ['sub_category', 'currency', 'country','disable_communication']

for ele in cat_features:

    ks[ele] = ks[ele].astype(str)



encoder = LabelEncoder()

# Apply the label encoder to each column

encoded = ks[cat_features].apply(encoder.fit_transform)

#encoded.head(5)
num_features = ['goal','train', 'launch_hour','launch_day', 'launch_month', 'launch_year','duration',

                'deadline_hour','deadline_day', 'deadline_month', 'deadline_year','outcome','name_len',

                'creator_freq',

                'sadness', 'joy', 'fear', 'disgust', 'anger', 'sentiment',

                'category-1 score', 'category-2 score', 'category-3 score','category-4 score', 'category-5 score']

for ele in num_features:

    ks[ele] = ks[ele].astype(float)
ks_new = ks[num_features].join(encoded)

train = ks_new[ks_new.train==1].drop(columns='train')

test = ks_new[ks_new.train==0].drop(columns=['train','outcome'])
from sklearn.model_selection import StratifiedKFold

import numpy as np



skf = StratifiedKFold(n_splits=5)

X = train.drop(['outcome'], axis=1)

y = train[['outcome']]
import lightgbm as lgb

import sklearn.metrics as metrics



param = {'num_leaves': 64, 'max_depth': 7, 'objective': 'binary','metric':['auc', 'binary_logloss']}

ans = 0

ans_test = np.array(test.shape[0]*[0], dtype=np.float64)



for train_index, test_index in skf.split(X,y):

    train_index = np.random.permutation(train_index)

    test_index = np.random.permutation(test_index)

            

    X_train, X_test,y_train, y_test = X.iloc[train_index,], X.iloc[test_index,],y.iloc[train_index,], y.iloc[test_index,]  

    lgbm_train = lgb.Dataset(X_train,y_train)

    lgbm_val = lgb.Dataset(X_test,y_test)

        

    bst = lgb.train(param, train_set=lgbm_train, valid_sets=[lgbm_train, lgbm_val], early_stopping_rounds=200, verbose_eval=800)

    ans += metrics.roc_auc_score(y_test, bst.predict(X_test))

    ans_test += bst.predict(test)

        

ans/=5

ans_test/=5

print(ans)
output = ks[ks.train==0][['id']]

output['Predicted'] = ans_test

output = output.rename({'id':"Id"})

#output.to_csv('answer.csv',index=False)
ks_solution = pd.read_csv('../input/2020caohackathon/solution.csv')

ks_solution = ks_solution.merge(output,left_on="Id",right_on="id",how='inner')

ks_solution
metrics.roc_auc_score(ks_solution['Expected'], ks_solution['Predicted'])