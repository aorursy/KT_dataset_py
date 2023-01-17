import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import gc

import seaborn as sns

from tqdm import tqdm_notebook

%matplotlib inline
Kaggle=1

if Kaggle==0:

    train=pd.read_csv("train.csv")

    test=pd.read_csv("test.csv")

    sample_sub=pd.read_csv("sample_submission.csv")

else:

    train=pd.read_csv("../input/train.csv")

    test=pd.read_csv("../input/test.csv")

    sample_sub=pd.read_csv("../input/sample_submission.csv")
print(f'Train has {train.shape[0]} rows and {train.shape[1]} columns' )

print(f'Test has {test.shape[0]} rows and {test.shape[1]} columns' )
train.head()
train['target'].value_counts()
from sklearn.model_selection import train_test_split,KFold, cross_val_score, GridSearchCV,StratifiedKFold,StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, roc_auc_score

import eli5

from eli5.sklearn import PermutationImportance

from sklearn.preprocessing import StandardScaler

import xgboost as xgb

import lightgbm as lgb

from tqdm import tqdm_notebook

import feather

from bayes_opt import BayesianOptimization
train_model=train.drop('ID_code',axis=1)

test_model=test.drop('ID_code',axis=1)
X=train_model.drop('target',axis=1)

Y=train_model['target']
train_x,valid_x,train_y,valid_y=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=100)
print(f'Training has {train_x.shape[0]} rows and {train_x.shape[1]} columns' )

print(f'Validation has {valid_x.shape[0]} rows and {valid_x.shape[1]} columns' )
train_y.value_counts()
valid_y.value_counts()
# folds = StratifiedKFold(n_splits=5,shuffle=True,random_state=40)
feature_name=[f for f in train_x.columns if f not in ['target']]

mean_auc=0.0

N_SPLITS=5
train_x.shape,train_y.shape
# train_x_sample=train_x.iloc[1:1000,]

# train_y_sample=train_y.iloc[1:1000]
# train_x_sample.shape,train_y_sample.shape
# def rf_model(**params):

#     params['min_samples_leaf']=int(params['min_samples_leaf'])

#     params['max_features']=int(params['max_features'])

#     params['max_depth']=int(params['max_depth'])

#     params['n_estimators']=int(params['n_estimators'])

   

    

#     test_pred=np.zeros(train_x.shape[0])

    

#     for n_folds,(train_idx,valid_idx) in enumerate(folds.split(train_x,train_y)):

#         x_train,x_valid=train_x.iloc[train_idx],train_x.iloc[valid_idx]

#         y_train,y_valid=train_y.iloc[train_idx],train_y.iloc[valid_idx]

#         clf=RandomForestClassifier(**params,random_state=100,n_jobs=-1,verbose=True)

#         clf.fit(x_train,y_train)

#         y_pred_proba=clf.predict_proba(x_valid)

        

#         test_pred[valid_idx]=clf.predict_proba(x_valid)[:,1]

        

#     gc.collect()

        

#     return roc_auc_score(y_valid,test_pred[valid_idx])

    
# params ={'n_estimators':(100,1000),

#           'max_depth':(10,100),

#           'min_samples_leaf':(1,10),

#          'max_features':(1,10)}
# bo = BayesianOptimization(rf_model, params)

# bo.maximize(init_points=5, n_iter=5)
# bo.max
rf_oof_preds=np.zeros(train_x.shape[0])

rf_sub_preds=np.zeros(test_model.shape[0])
folds=StratifiedShuffleSplit(n_splits=5,random_state=100)


auc_score=[]

importance=pd.DataFrame()

%time

for n_folds,(train_idx,valid_idx) in enumerate(folds.split(train_x,train_y)):

    x_train,x_valid=train_x.iloc[train_idx],train_x.iloc[valid_idx]

    y_train,y_valid=train_y.iloc[train_idx],train_y.iloc[valid_idx]

    clf=RandomForestClassifier(n_estimators=720 ,max_depth= 50,min_samples_leaf=9 ,max_features=2 ,n_jobs=-1,random_state=100,verbose=True)

    clf.fit(x_train,y_train)

    y_preds_proba=clf.predict_proba(x_valid)

    rf_oof_preds[valid_idx]=y_preds_proba[:,1]

    rf_sub_preds=clf.predict_proba(test_model[feature_name])[:,1]/folds.n_splits

    auc_score.append(roc_auc_score(y_valid,rf_oof_preds[valid_idx]))

    

    print("\n {} fold ROC AUC Score is : {}".format(n_folds+1,roc_auc_score(y_valid,rf_oof_preds[valid_idx])))

    

    importance['feature']=feature_name

    importance['gini']=clf.feature_importances_

    importance['fold']=n_folds+1

    

print("\n Average ROC Score is {}",np.mean(auc_score))

    

    

    
### https://www.kaggle.com/gpreda/santander-eda-and-prediction

lgb_oof_preds=np.zeros(train_x.shape[0])

lgb_sub_preds=np.zeros(test_model.shape[0])

auc_valid=[]

param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.4,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.05,

    'learning_rate': 0.01,

    'max_depth': -1,  

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': 1

}
importances=pd.DataFrame()

for fold_idx, (train_ids, valid_ids) in enumerate(folds.split(train_x,train_y)):

    # Split traninig data set.

    trn_data = lgb.Dataset(train_x.iloc[train_ids],label=train_y.iloc[train_ids])

    val_data = lgb.Dataset(train_x.iloc[valid_ids],label=train_y.iloc[valid_ids])

    ## Building the model:

    num_rounds=10000

    clf = lgb.train(param, trn_data, num_rounds, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)

    # Train estimator.

    

    # Prediction and evaluation on validation data set.

    lgb_oof_preds[valid_ids] = clf.predict(train_x.iloc[valid_ids],num_iteration=clf.best_iteration)

    # Set feature importances.

    imp_df = pd.DataFrame()

    imp_df['feature'] = feature_name

    imp_df['gain'] = clf.feature_importance()

    imp_df['fold'] = fold_idx + 1

    importances = pd.concat([importances, imp_df], axis=0, sort=False)

    

    

    # Prediction of testing data set.

    lgb_sub_preds += clf.predict(test_model[feature_name],num_iteration=clf.best_iteration)/ folds.n_splits

    

    

    

    gc.collect()

print("Mean AUC: %.5f" % (roc_auc_score(train_y,lgb_oof_preds)))
# ### Taken from Oliver's awesome kernel - 



# def display_importances(feature_importance_df_):

#     # Plot feature importances

#     cols = feature_importance_df_[["feature", "gain"]].groupby("feature").mean().sort_values(

#         by="gain", ascending=False)[:50].index

    

#     best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    

#     plt.figure(figsize=(8,10))

#     sns.barplot(x="gain", y="feature", 

#                 data=best_features.sort_values(by="gain", ascending=False))

#     plt.title('LightGBM Features (avg over folds)')

#     plt.tight_layout()

#     #plt.savefig('lgbm_importances.png')

#  display_importances(importances)
## Blending and submitting



sample_submission = pd.DataFrame({"ID_code":test["ID_code"].values})

sample_submission["target"] = (0.4*rf_sub_preds)+(0.6*lgb_sub_preds)

sample_submission.to_csv("blend_submission.csv", index=False)