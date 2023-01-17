!pip install xgboost



import numpy as np

import pandas as pd



from sklearn.datasets import load_iris

import xgboost as xgb

from sklearn.metrics import accuracy_score,roc_auc_score, f1_score

 

from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold, KFold,cross_val_score, GridSearchCV
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
train= pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test= pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

sub   = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

train.head()



train.target.value_counts()
train['sex'] = train['sex'].fillna('na')

train['age_approx'] = train['age_approx'].fillna(0)

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')



test['sex'] = test['sex'].fillna('na')

test['age_approx'] = test['age_approx'].fillna(0)

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')
train['sex'] = train['sex'].astype("category").cat.codes +1

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].astype("category").cat.codes +1

train.head()
train['target'].value_counts()
test['sex'] = test['sex'].astype("category").cat.codes +1

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].astype("category").cat.codes +1

test.head()
train['patient_id'].shape, train['patient_id'].nunique(), 
test['patient_id'].shape, test['patient_id'].nunique(), 


X = train[['sex', 'age_approx','anatom_site_general_challenge']]

y = train['target']





test_X = test[['sex', 'age_approx','anatom_site_general_challenge']]



def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, rounds=500, dep=8, eta=0.05):

    params = {}

    params["objective"] = "binary:logistic"

    params['eval_metric'] = 'auc'

    params["eta"] = 0.09

    params["subsample"] = 0.9

    params["min_child_weight"] = 1

    params["colsample_bytree"] = 0.9

    params["max_depth"] = 4

    params["silent"] = 1

    params["seed"] = seed_val

    params["n_estimators"] = 500

    params["reg_alpha"] = 0.05



    params["gamma"] = 1

    num_rounds = rounds



    plst = list(params.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)



    xgtest = xgb.DMatrix(test_X, label=test_y)

    watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=200, verbose_eval=500)





    pred_test_y = model.predict(xgtest, ntree_limit=model.best_ntree_limit)

    pred_test_y2 = model.predict(xgb.DMatrix(test_X2), ntree_limit=model.best_ntree_limit)

    

    loss = roc_auc_score(test_y, pred_test_y)

    return pred_test_y, loss, pred_test_y2, model
cv_scores = []

pred_test_full = 0





# kf = GroupKFold(n_splits=5)

# kf = StratifiedKFold(n_splits=5, shuffle=True)

# kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=30)

kf = KFold(n_splits=10, shuffle=True, random_state=30)



for dev_index, val_index in kf.split(X,y):

    dev_X, val_X = X.loc[dev_index,:], X.loc[val_index,:]

    dev_y, val_y = y[dev_index], y[val_index]



    

    pred_val, loss, pred_test, model = runXGB(dev_X, dev_y, val_X, val_y, test_X)

    

    f1_scores.append((f1_score(val_y, np.where(pred_val >=0.50,1,0), average='binary')))

    pred_test_full +=pred_test

    

    cv_scores.append(loss)

    print(cv_scores)



pred_test_full /=10.



print('Avg AUC Score :',sum(cv_scores)/10)

  
# Avg AUC Score : 0.697888987983679
sub.target = pred_test_full

sub.head()
sub.to_csv('submission.csv',index=False)