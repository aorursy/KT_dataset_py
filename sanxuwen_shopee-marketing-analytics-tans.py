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
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
import datetime as dt
import pickle
df_train = pd.read_csv('../input/student-shopee-code-league-marketing-analytics/train.csv')
df_users = pd.read_csv('../input/student-shopee-code-league-marketing-analytics/users.csv')
df_test = pd.read_csv('../input/student-shopee-code-league-marketing-analytics/test.csv')
df_sub = pd.read_csv('../input/student-shopee-code-league-marketing-analytics/sample_submission_0_1.csv')
print('The train shape:',df_train.shape)
print('Users shape:', df_users.shape)
df_merge = df_train.merge(df_users,on = 'user_id',how = 'left')
df_test = df_test.merge(df_users,on = 'user_id',how = 'left')

print('The train shape:',df_merge.shape)
print('Users shape:', df_test.shape)
#attr_1 and attr_2 fillwith 
df_merge.isna().mean()
intersect_userid = set(df_merge['user_id']).intersection(df_test['user_id'])
df_merge['grass_date'] = df_merge['grass_date'].astype('datetime64')
df_merge['dayofweek'] = df_merge['grass_date'].dt.dayofweek
# df_merge['is_weekend'] = df_merge['dayofweek'].apply(lambda x: 1 if x>=5 else 0)
# df_merge['grass_day'] = df_merge['grass_date'].dt.day

df_test['grass_date'] = df_test['grass_date'].astype('datetime64')
df_test['dayofweek'] = df_test['grass_date'].dt.dayofweek
# df_test['is_weekend'] = df_test['dayofweek'].apply(lambda x: 1 if x>=5 else 0)
# df_test['grass_day'] = df_test['grass_date'].dt.day


df_merge = df_merge.drop(columns = ['grass_date'])
df_test = df_test.drop(columns = ['grass_date'])
df_merge['last_open_day'] = df_merge['last_open_day'].replace(to_replace ="Never open", value =-999) 
df_merge['last_checkout_day'] = df_merge['last_checkout_day'].replace(to_replace ="Never checkout", value = -999) 
df_merge['last_login_day'] = df_merge['last_login_day'].replace(to_replace ="Never login", value = -999) 

df_merge[['last_open_day','last_checkout_day','last_login_day']] = df_merge[['last_open_day','last_checkout_day','last_login_day']].astype(int)




df_test['last_open_day'] = df_test['last_open_day'].replace(to_replace ="Never open", value =-999) 
df_test['last_checkout_day'] = df_test['last_checkout_day'].replace(to_replace ="Never checkout", value =-999) 
df_test['last_login_day'] = df_test['last_login_day'].replace(to_replace ="Never login", value =-999) 

df_test[['last_open_day','last_checkout_day','last_login_day']] = df_merge[['last_open_day','last_checkout_day','last_login_day']].astype(int)
df_merge[['country_code','domain']] = df_merge[['country_code','domain']].astype(str)
df_test[['country_code','domain']] = df_test[['country_code','domain']].astype(str)
df_merge = pd.get_dummies(df_merge)
df_test = pd.get_dummies(df_test)
features_train = df_merge.drop(columns = ['open_flag','row_id']).copy()
target_train = df_merge['open_flag'].copy()

print('Features train shape', features_train.shape)

from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import ClusterCentroids
from sklearn import preprocessing

scaler = MinMaxScaler()
temp = df_merge.drop(columns = ['open_flag','row_id','user_id']).fillna(df_merge.drop(columns = ['open_flag','row_id','user_id']).median())

features_train = pd.DataFrame(scaler.fit_transform(temp),columns = temp.columns)
target_train = df_merge['open_flag']



# # Run this to use tomeklinks
tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(features_train, df_merge['open_flag'])
features_train = X_tl
target_train = y_tl


# print(features_train.shape)


data_dmatrix = xgb.DMatrix(data=features_train,label=target_train)
THRESHOLD = 0.4


def evalmcc(preds, dtrain):
    labels = dtrain.get_label()
    return 'MCC', matthews_corrcoef(labels, preds > THRESHOLD)


def evalmcc_min(preds, dtrain):
    labels = dtrain.get_label()
    return 'MCC', matthews_corrcoef(labels, preds > THRESHOLD)


xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.5, 
    'subsample': 0.5,  
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'max_depth': 7, 
    'min_child_weight': 5,
    'scale_pos_weight': 1,
}

res = xgb.cv(xgb_params, data_dmatrix, num_boost_round=1000, nfold=5, seed=0, stratified=True,
            early_stopping_rounds=25, verbose_eval=1, show_stdv=True, feval=evalmcc, maximize=True)


xg_cls = xgb.train(params=xgb_params, dtrain=data_dmatrix,num_boost_round=103,maximize=True)
import matplotlib.pyplot as plt
xgb.plot_importance(xg_cls)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()
temp = df_test.drop(columns = ['row_id','user_id']).fillna(df_test.drop(columns = ['row_id','user_id']).median()) 
test_X = pd.DataFrame(scaler.fit_transform(temp),columns = temp.columns) #Case scale
pred_xgboost = xg_cls.predict(xgb.DMatrix(test_X))
pred_xgboost
pred_xgboost[39]
df_submission = pd.DataFrame({'row_id':[i for i in range(len(df_test))],'open_flag':pred_xgboost})
df_submission['open_flag'] = df_submission['open_flag'].apply(lambda x : 1 if x>= 0.4 else 0)
df_submission.to_csv('submission_fillage_(0.4)_tomek_median.csv',index = False)
df_submission['open_flag'].value_counts(normalize = True)
from numba import jit

# @jit
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)

# @jit
def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    print(idx)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        print(score, best_mcc)
        plt.plot(mccs)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc

## 5 fold CV
from catboost import CatBoostClassifier, cv, Pool
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
import matplotlib.pyplot as plt
skf = StratifiedKFold(n_splits=10, random_state=2021, shuffle=True)



def evalerror_mat_coeff(y_true, p_pred_1):
    # convert probability (y=1) to label with threshold 0.5
    y_pred = [1 if pred>=THRESHOLD else 0 for pred in p_pred_1]
    return 'MCC', matthews_corrcoef(y_true, y_pred), True

#Run this for no polynomial
temp = df_test.drop(columns = ['row_id','user_id']).fillna(df_test.drop(columns = ['row_id','user_id']).median()) 
X_test = pd.DataFrame(scaler.fit_transform(temp),columns = temp.columns)



best_epochs = []
fold_scores = []
test_pred_cv = []
best_cutoff = []
vote_predictions = []
for i,(train_index, val_index) in enumerate(skf.split(features_train, target_train)):
    print(f"<------------------------------------- Fold: {i+1} ------------------------------------->")
    print("TRAIN:", train_index, "TEST:", val_index)
    X_train, X_val = features_train.iloc[train_index], features_train.iloc[val_index]
    y_train, y_val = target_train.iloc[train_index], target_train.iloc[val_index]
    
    # Best tuned
    params = {'loss_function':'Logloss', # objective function
          'eval_metric':'MCC:hints=skip_train~false', 
          'verbose': 200, # output to stdout info about training process every 200 iterations
          'random_seed': 2020,
          'iterations': 3000,
          'depth': 8, 
          'l2_leaf_reg': 3,
          'learning_rate': 0.05,
         }  
    
    
    cbc_1 = CatBoostClassifier(**params,class_weights=[1, 1.75])
    
    
    cbc_1.fit(X_train, y_train, 
          eval_set=(X_val, y_val), 
          use_best_model=True, 
          early_stopping_rounds=50,
          plot=True)
    val_pred = cbc_1.predict(X_val)
    
    test_predictions = cbc_1.predict_proba(X_test)
    test_pred_cv.append(test_predictions)
    

    
    val_pred_proba = cbc_1.predict_proba(X_val)
    
    best_proba, best_mcc, y_pred = eval_mcc(np.array(y_val), np.array([pred[1] for pred in val_pred_proba]), True)
    
    print('Best Thres:',best_proba)
    best_cutoff.append(best_proba)
    print('Best MCC',best_mcc)
    THRESHOLD = best_proba
    
    
    fold_score = matthews_corrcoef(y_val, val_pred)
    print(f"MCC score of fold no adjust threshold {i+1}:{fold_score}")
    fold_scores.append(fold_score)
    
    cv_vote_pred = [1 if p[1]>=best_proba else 0 for p in test_predictions]
    vote_predictions.append(cv_vote_pred)
 

np.mean(fold_scores)
np.mean(best_cutoff)
sum(test_pred_cv)/10
final_preds_cat = []
preds_cat = sum(test_pred_cv)/10
for pred in preds_cat:
    final_preds_cat.append(pred[1])
final_preds_cat
df_submission = pd.DataFrame({'row_id':[i for i in range(len(df_test))],'open_flag':final_preds_cat})
df_submission['open_flag'] = df_submission['open_flag'].apply(lambda x : 1 if x>= 0.53 else 0)
df_submission.to_csv('submission_fillage_(0.53)_catboost.csv',index = False)
df_submission['open_flag'].value_counts(normalize = True)
