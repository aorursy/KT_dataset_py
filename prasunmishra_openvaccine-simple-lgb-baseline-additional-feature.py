import gc

import os

import random



import lightgbm as lgb

import numpy as np

import pandas as pd

import seaborn as sns

import itertools



from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

from sklearn.cluster import KMeans



sns.set(style='darkgrid')

SEEDS = 42
def rmse(y_true, y_pred):

    return (mean_squared_error(y_true, y_pred))** .5
# treemodel_wrapper

class TreeModel:

    def __init__(self, model_type):

        self.model_type = model_type

        self.tr_data = None

        self.vl_data = None

        self.model = None

    

    def train(self, params, train_x, train_y, valid_x=None, valid_y=None, num_round=None, early_stopping=None, verbose=None):

        if self.model_type == 'lgb':

            self.tr_data = lgb.Dataset(train_x, label=train_y)

            self.vl_data = lgb.Dataset(valid_x, label=valid_y)

            self.model = lgb.train(params, self.tr_data, valid_sets=[self.tr_data, self.vl_data],

                                   num_boost_round=num_round, early_stopping_rounds=early_stopping,verbose_eval=verbose)

            

        if self.model_type == 'rf_reg':

            self.train_x = train_x

            self.train_y = train_y

            self.model = RandomForestRegressor(**params).fit(self.train_x, self.train_y)

            

        if self.model_type == 'xgb':

            self.tr_data = xgb.DMatrix(train_x, train_y)

            self.vl_data = xgb.DMatrix(valid_x, valid_y)

            self.model = xgb.train(params, self.tr_data, num_boost_round=num_round,

                                   evals=[(self.tr_data, 'train'), (self.vl_data, 'val')], 

                                   verbose_eval=verbose, early_stopping_rounds=early_stopping)

            

        if self.model_type == 'cat':

            params['num_boost_round'] = num_round

            self.cat_cols = list(train_x.select_dtypes(include='object').columns)

            self.tr_data = Pool(train_x, train_y, cat_features=self.cat_cols)

            self.vl_data = Pool(valid_x, valid_y, cat_features=self.cat_cols)

            self.model = CatBoost(params).fit(self.tr_data, eval_set=self.vl_data,

                                                early_stopping_rounds=early_stopping, verbose=verbose, use_best_model=True)

            

            return self.model

            

    

    def predict(self,X):

        if self.model_type == 'lgb':

            return self.model.predict(X, num_iteration=self.model.best_iteration)

        

        if self.model_type == 'rf_reg':

            return self.model.predict(X)

        

        if self.model_type == 'xgb':

            X_DM = xgb.DMatrix(X)

            return self.model.predict(X_DM)

        

        if self.model_type == 'cat':

            X_pool = Pool(X, cat_features=self.cat_cols)

            return self.model.predict(X_pool)

    

    @property

    def feature_names_(self):

        if self.model_type == 'lgb':

            return self.model.feature_name()

        

        if self.model_type == 'rf_reg':

            return self.train_x.columns

        

        if self.model_type == 'xgb':

            return list(self.model.get_score(importance_type='gain').keys())

        

        if self.model_type == 'cat':

            return self.model.feature_names_

    

    @property

    def feature_importances_(self):

        if self.model_type == 'lgb':

            return self.model.feature_importance(importance_type='gain')

        

        if self.model_type == 'rf_reg':

            return self.model.feature_importances_

        

        if self.model_type == 'xgb':

            return list(self.model.get_score(importance_type='gain').values())

        

        if self.model_type == 'cat':

            return self.model.feature_importances_
PATH = '../input/stanford-covid-vaccine/'

train = pd.read_json(PATH+'train.json',lines=True)

test = pd.read_json(PATH+'test.json', lines=True)

submission = pd.read_csv(PATH+'sample_submission.csv')
train[train['signal_to_noise'] > 1].shape
train[train['SN_filter'] == 1].shape
train[train['SN_filter'] == 1].head(5)
train = train[train['SN_filter'] == 1] 

train.shape
print(train.sequence.values)
#Additional features

#Basic ideas is that GC (or CG) is strongest pair, AU (or UA) is weaker and GU(or UG) is the weakest. hence counting the pairs occurance and adding

#as a feature will provide better signals to LGBM

train['GCcount1']=train['sequence'].map(lambda x: x.count('GC'))

train['GCcount2']=train['sequence'].map(lambda x: x.count('CG'))



train['AUcount1']=train['sequence'].map(lambda x: x.count('AU'))

train['AUcount2']=train['sequence'].map(lambda x: x.count('UA'))



train['GUcount1']=train['sequence'].map(lambda x: x.count('GU'))

train['GUcount2']=train['sequence'].map(lambda x: x.count('UG'))



train['GCcount']=train['GCcount1']+train['GCcount2']

train['AUcount']=train['AUcount1']+train['AUcount2']

train['GUcount']=train['GUcount1']+train['GUcount2']
train = train.drop(['GCcount1','GCcount2','AUcount1','AUcount2','GUcount1','GUcount2'], axis=1)

train.head(3)
test.columns

train_data = []

for mol_id in train['id'].unique():

    sample_data = train.loc[train['id'] == mol_id]

    sample_seq_length = sample_data.seq_length.values[0]

    

    for i in range(68):

        sample_dict = {'id' : sample_data['id'].values[0],

                       'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),

                       'sequence' : sample_data['sequence'].values[0][i],

                       'structure' : sample_data['structure'].values[0][i],

                       'predicted_loop_type' : sample_data['predicted_loop_type'].values[0][i],

                       'reactivity' : sample_data['reactivity'].values[0][i],

                       'reactivity_error' : sample_data['reactivity_error'].values[0][i],

                       'deg_Mg_pH10' : sample_data['deg_Mg_pH10'].values[0][i],

                       'deg_error_Mg_pH10' : sample_data['deg_error_Mg_pH10'].values[0][i],

                       'deg_pH10' : sample_data['deg_pH10'].values[0][i],

                       'deg_error_pH10' : sample_data['deg_error_pH10'].values[0][i],

                       'deg_Mg_50C' : sample_data['deg_Mg_50C'].values[0][i],

                       'deg_error_Mg_50C' : sample_data['deg_error_Mg_50C'].values[0][i],

                       'deg_50C' : sample_data['deg_50C'].values[0][i],

                       'deg_error_50C' : sample_data['deg_error_50C'].values[0][i],

                       'GCcount':sample_data['GCcount'].values[0],

                       'AUCcount':sample_data['AUcount'].values[0],

                       'GUcount':sample_data['GUcount'].values[0]}

        

        

        shifts = [1,2,3,4,5]

        shift_cols = ['sequence', 'structure', 'predicted_loop_type']

        #shift_cols = ['sequence']

        for shift,col in itertools.product(shifts, shift_cols):

            if i - shift >= 0:

                sample_dict['b'+str(shift)+'_'+col] = sample_data[col].values[0][i-shift]

            else:

                sample_dict['b'+str(shift)+'_'+col] = -1

            

            if i + shift <= sample_seq_length - 1:

                sample_dict['a'+str(shift)+'_'+col] = sample_data[col].values[0][i+shift]

            else:

                sample_dict['a'+str(shift)+'_'+col] = -1

        

        

        train_data.append(sample_dict)

train_data = pd.DataFrame(train_data)

train_data.head()
#Additional features

#Basic ideas is that GC (or CG) is strongest pair, AU (or UA) is weaker and GU(or UG) is the weakest. hence counting the pairs occurance and adding

#as a feature will provide better signals to LGBM

test['GCcount1']=test['sequence'].map(lambda x: x.count('GC'))

test['GCcount2']=test['sequence'].map(lambda x: x.count('CG'))



test['AUcount1']=test['sequence'].map(lambda x: x.count('AU'))

test['AUcount2']=test['sequence'].map(lambda x: x.count('UA'))



test['GUcount1']=test['sequence'].map(lambda x: x.count('GU'))

test['GUcount2']=test['sequence'].map(lambda x: x.count('UG'))



test['GCcount']=test['GCcount1']+test['GCcount2']

test['AUcount']=test['AUcount1']+test['AUcount2']

test['GUcount']=test['GUcount1']+test['GUcount2']



test = test.drop(['GCcount1','GCcount2','AUcount1','AUcount2','GUcount1','GUcount2'], axis=1)

test.head(3)
test_data = []

for mol_id in test['id'].unique():

    sample_data = test.loc[test['id'] == mol_id]

    sample_seq_length = sample_data.seq_length.values[0]

    for i in range(sample_seq_length):

        sample_dict = {'id' : sample_data['id'].values[0],

                       'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),

                       'sequence' : sample_data['sequence'].values[0][i],

                       'structure' : sample_data['structure'].values[0][i],

                       'predicted_loop_type' : sample_data['predicted_loop_type'].values[0][i],

                       'GCcount':sample_data['GCcount'].values[0],

                       'AUCcount':sample_data['AUcount'].values[0],

                       'GUcount':sample_data['GUcount'].values[0]}

        

        

        shifts = [1,2,3,4,5]

        shift_cols = ['sequence', 'structure', 'predicted_loop_type']

        #shift_cols = ['sequence']

        for shift,col in itertools.product(shifts, shift_cols):

            if i - shift >= 0:

                sample_dict['b'+str(shift)+'_'+col] = sample_data[col].values[0][i-shift]

            else:

                sample_dict['b'+str(shift)+'_'+col] = -1

            

            if i + shift <= sample_seq_length - 1:

                sample_dict['a'+str(shift)+'_'+col] = sample_data[col].values[0][i+shift]

            else:

                sample_dict['a'+str(shift)+'_'+col] = -1

        

        test_data.append(sample_dict)

test_data = pd.DataFrame(test_data)

test_data.head()
# label_encoding

sequence_encmap = {'A': 0, 'G' : 1, 'C' : 2, 'U' : 3}

structure_encmap = {'.' : 0, '(' : 1, ')' : 2}

looptype_encmap = {'S':0, 'E':1, 'H':2, 'I':3, 'X':4, 'M':5, 'B':6}



enc_targets = ['sequence', 'structure', 'predicted_loop_type']

enc_maps = [sequence_encmap, structure_encmap, looptype_encmap]



for t,m in zip(enc_targets, enc_maps):

    for c in [c for c in train_data.columns if t in c]:

        train_data[c] = train_data[c].astype(str).replace(m)

        test_data[c] = test_data[c].astype(str).replace(m)
print(train_data.shape)

print(train_data.dtypes)

train_data.head(3)
print(test_data.shape)

test_data.head(3)
not_use_cols = ['id', 'id_seqpos']

features = [c for c in test_data.columns if c not in not_use_cols]

targets = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']



    

for f in features:

    if test_data[f].dtype == 'object':

        train_data[f]= train_data[f].astype(str).astype(int)

        test_data[f]= test_data[f].astype(str).astype(int)

len(features)
FOLD_N = 5

gkf = GroupKFold(n_splits=FOLD_N)
"""

params = {'objective': 'regression',

          'boosting': 'gbdt',

          'metric': 'rmse',

          'learning_rate': 0.01,

          'seed' : SEEDS}

"""

n_fold =5

params ={'boosting': 'gbdt',

        'objective': 'regression',

        'metric': 'rmse',

        'learning_rate': 0.04,

        'subsample': 0.7,

        'max_depth': -1,

        'num_leaves': 81,

        'colsample_bytree': 0.8,

        'verbose': -1,

        'seed':int(2**n_fold),

        'bagging_seed':int(2**n_fold)

        }
feature_importances = pd.DataFrame()

result = {}

oof_df = pd.DataFrame(train_data.id_seqpos)



for target in targets:

    oof = pd.DataFrame()

    preds = np.zeros(len(test_data))

    scores = 0.0

    

    for n, (tr_idx, vl_idx) in enumerate(gkf.split(train_data[features], train_data['reactivity'], train_data['id'])):

        tr_x, tr_y = train_data[features].iloc[tr_idx], train_data[target].iloc[tr_idx]

        vl_x, vl_y = train_data[features].iloc[vl_idx], train_data[target].iloc[vl_idx]

        vl_id = train_data['id_seqpos'].iloc[vl_idx]



        model = TreeModel(model_type='lgb')

        model.train(params, tr_x, tr_y, vl_x, vl_y,

                    num_round=20000, early_stopping=100,verbose=1000)



        fi_tmp = pd.DataFrame()

        fi_tmp['feature'] = model.feature_names_

        fi_tmp['importance'] = model.feature_importances_

        fi_tmp['fold'] = n

        fi_tmp['target'] = target

        feature_importances = feature_importances.append(fi_tmp)



        vl_pred = model.predict(vl_x)

        score = rmse(vl_y, vl_pred)

        scores += score / FOLD_N

        print(f'score : {score}')



        oof = oof.append(pd.DataFrame({'id_seqpos':vl_id, target:vl_pred}))



        pred = model.predict(test_data[features])

        preds += pred / FOLD_N

    

    oof_df = oof_df.merge(oof, on='id_seqpos', how='inner')

    submission[target] = preds

    

    print(f'{target}_rmse : {scores}')

    result[target] = scores
display(result)

display(f'total : {np.mean(list(result.values()))}')
# feature_importances

for target in targets:

    tmp = feature_importances[feature_importances.target==target]

    order = list(tmp.groupby('feature').mean().sort_values('importance', ascending=False).index)



    plt.figure(figsize=(10, 5))

    sns.barplot(x="importance", y="feature", data=tmp, order=order)

    plt.title(target)

    plt.tight_layout()
oof_df.head()
submission.head()
display(oof_df.shape)

display(submission.shape)
oof_df.to_csv('oof_df.csv', index=False)

submission.to_csv('submission_lgb.csv', index=False)
sub_col= submission.columns
