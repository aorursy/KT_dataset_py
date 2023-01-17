# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

print (os.listdir('../input/'))

# Any results you write to the current directory are saved as output.
%%time

from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,RepeatedKFold

from sklearn.linear_model import BayesianRidge



def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    """

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling

    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric

    """

    maes = (y_true-y_pred).abs().groupby(types).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()



train = pd.read_csv('../input/pmp-oof/final_train_oof_pmp.csv')

test = pd.read_csv('../input/pmp-oof/final_test_oof_pmp.csv')

drop_features=['id','type','scalar_coupling_constant','molecule_name'

              ]#

feats = [f for f in train.columns if f not in drop_features]

print ('features:',feats)



n_splits= 5

folds = GroupKFold(n_splits=n_splits)

oof_preds = np.zeros((train.shape[0]))

sub_preds = np.zeros((test.shape[0]))



for t in train['type'].unique():

    train_t = train[train['type']==t].reset_index(drop=True)

    idx = train[train['type']==t].index

    test_t = test[test['type']==t].reset_index(drop=True)

    idx_test = test[test['type']==t].index

    cv_list = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_t,groups=train_t['molecule_name'])):

        train_x, train_y = train_t[feats].iloc[train_idx], train_t['scalar_coupling_constant'].iloc[train_idx]

        valid_x, valid_y = train_t[feats].iloc[valid_idx], train_t['scalar_coupling_constant'].iloc[valid_idx] 

        valid_type = train_t['type'].iloc[valid_idx] 

        train_x = train_x.values

        valid_x = valid_x.values

        

        clf = BayesianRidge(

                            n_iter=1000,#50

                            tol=0.1,#0.01

                            normalize=False)

        clf.fit(train_x, train_y)

    

        oof_preds[idx[valid_idx]] = clf.predict(valid_x)

        oof_cv = group_mean_log_mae(y_true=valid_y, 

                              y_pred=oof_preds[idx[valid_idx]], 

                              types=valid_type)



        cv_list.append(oof_cv)

        sub_preds[idx_test]  += clf.predict(test_t[feats].values) / folds.n_splits

    print ('type=' + str(t),cv_list) 



train['stacking'] = oof_preds

test['scalar_coupling_constant'] = sub_preds



for t in train['type'].unique():

    train_t = train[train['type']==t]

    oof_type = group_mean_log_mae(y_true=train_t['scalar_coupling_constant'], 

                            y_pred=train_t['stacking'], 

                            types=train_t['type'])        

    print (t,oof_type)



oof_full = group_mean_log_mae(y_true=train['scalar_coupling_constant'], 

                            y_pred=oof_preds, 

                            types=train['type'])        

print ('All type',oof_full)



train[['id','stacking']].to_csv('train_stacking.csv',index=False)

test[['id','scalar_coupling_constant']].to_csv('test_stacking.csv',index=False)