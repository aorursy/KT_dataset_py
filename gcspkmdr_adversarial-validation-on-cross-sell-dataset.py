import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

import xgboost as xgb

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder
train_data = pd.read_csv('../input/avcrosssell/train.csv')



train_data.drop(['id', 'Response'], axis = 1, inplace = True)



test_data = pd.read_csv('../input/avcrosssell/test.csv')



test_data.drop(['id'], axis = 1, inplace = True)
train_data['istrain'] = 1



test_data['istrain'] = 0



combined_data = pd.concat([train_data, test_data], axis = 0)
df_numeric = combined_data.select_dtypes(exclude=['object'])



df_obj = combined_data.select_dtypes(include=['object']).copy()

    

for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]

    

combined_data = pd.concat([df_numeric, df_obj], axis=1)



y = combined_data['istrain']



combined_data.drop('istrain', axis = 1, inplace = True)

skf = StratifiedShuffleSplit(n_splits = 5, random_state = 44,test_size =0.3)

xgb_params = {

        'learning_rate': 0.1, 'max_depth': 6,'subsample': 0.9,

        'colsample_bytree': 0.9,'objective': 'binary:logistic',

        'n_estimators':100, 'gamma':1,

        'min_child_weight':4

        }   

clf = xgb.XGBClassifier(**xgb_params, seed = 10)     
for train_index, test_index in skf.split(combined_data, y):

       

        x0, x1 = combined_data.iloc[train_index], combined_data.iloc[test_index]

        

        y0, y1 = y.iloc[train_index], y.iloc[test_index]        

        

        print(x0.shape)

        

        clf.fit(x0, y0, eval_set=[(x1, y1)],

               eval_metric='logloss', verbose=False,early_stopping_rounds=10)

                

        prval = clf.predict_proba(x1)[:,1]

        

        print(roc_auc_score(y1,prval))