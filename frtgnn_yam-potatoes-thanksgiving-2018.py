import os

import warnings

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.model_selection import StratifiedKFold, KFold



%matplotlib inline

print(os.listdir("../input"))

warnings.filterwarnings('ignore')
train  = pd.read_json('../input/train.json')

test   = pd.read_json('../input/test.json')

sample = pd.read_csv('../input/sample_submission.csv')
train.shape, test.shape
train.head()
train.info()

print('_' * 60, "\n")

test.info()
a = train['vid_id'].unique()

b = test['vid_id'].unique()



any (i in a for i in b)
plt.figure(figsize=(10, 6))



sns.despine()

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})



sns.distplot(train['start_time_seconds_youtube_clip'],label='Start')

sns.distplot(train['end_time_seconds_youtube_clip'],label='End')

plt.title('Train Data Start & End Distribution')

plt.legend(loc="upper right")

plt.xlabel('Start & End Time for the clips')

plt.ylabel('Distribution')
plt.figure(figsize=(10, 6))



sns.despine()

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})



sns.distplot(train['start_time_seconds_youtube_clip'],label='Train')

sns.distplot(test['start_time_seconds_youtube_clip'],label='Test')

plt.title('Train & Test Data Start Distributions')

plt.legend(loc="upper right")

plt.xlabel('Start Time for the clips')

plt.ylabel('Distribution')
plt.figure(figsize=(10, 6))



sns.despine()

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})



sns.distplot(train['end_time_seconds_youtube_clip'],label='Train')

sns.distplot(test['end_time_seconds_youtube_clip'],label='Test')

plt.title('Train & Test Data End Distributions')

plt.legend(loc="upper right")

plt.xlabel('End Time for the clips')

plt.ylabel('Distribution')
plt.figure(figsize=(8, 8))

train['is_turkey'].value_counts().plot(kind='bar')

plt.title('Train & Test Data End Distributions')

plt.xlabel('Target Labels')

plt.ylabel('Count')
# got this two funcs from Tee Ming Yi, thanks!

#https://www.kaggle.com/teemingyi/turkey-competition

def create_df(data, i):

    df = pd.DataFrame([x for x in data.audio_embedding.iloc[i]])

    df['vid_id'] = data.vid_id.iloc[i]

    return df
def create_df_test(data, i):

    df = pd.DataFrame([x for x in data.audio_embedding.iloc[i]])

    df['vid_id'] = data.vid_id.iloc[i]

    return df
vid_train = []

for i in range(len(train.index)):

    vid_train.append(create_df(train, i))

    

vid_train_flatten = pd.concat(vid_train)  

vid_train_flatten.columns = ['feature_'+str(x) for x in vid_train_flatten.columns[:128]] + ['vid_id']



#



vid_test = []

for i in range(len(test.index)):

    vid_test.append(create_df_test(test, i))

    

vid_test_flatten = pd.concat(vid_test)  

vid_test_flatten.columns = ['feature_'+str(x) for x in vid_test_flatten.columns[:128]] + ['vid_id']
vid_train_flatten.shape, vid_test_flatten.shape
vid_train_flatten.info()

print('_' * 60, "\n")

vid_test_flatten.info()
df_train = pd.merge(train,vid_train_flatten, on = 'vid_id')

df_test  = pd.merge(test, vid_test_flatten , on = 'vid_id')



df_train = df_train.drop(['audio_embedding'],axis=1)

df_test  = df_test.drop(['audio_embedding'], axis=1)
df_train.shape, df_test.shape
abs(df_train.corr())['is_turkey'].sort_values(ascending=False)[:10]
high_corr = pd.DataFrame(abs(df_train.corr()[:10]))

high_corr_square = high_corr[high_corr.columns[:10].tolist()]



sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2})

plt.figure(figsize = (12,12))

sns.heatmap(high_corr_square,linecolor ='white',linewidths=1,annot=True)
df_train['duration'] = df_train['end_time_seconds_youtube_clip']-df_train['start_time_seconds_youtube_clip']

df_test['duration'] = df_test['end_time_seconds_youtube_clip']-df_test['start_time_seconds_youtube_clip']
train_columns = df_train.columns

test_columns  = df_test.columns



df_train['all_feature_mean'] = df_train[train_columns[4:131]].mean(axis=1)

df_test['all_feature_mean']  = df_test[test_columns[3:130]].mean(axis=1)



df_train['all_feature_median'] = df_train[train_columns[4:131]].median(axis=1)

df_test['all_feature_median']  = df_test[test_columns[3:130]].median(axis=1)



df_train['all_feature_min'] = df_train[train_columns[4:131]].min(axis=1)

df_test['all_feature_min']  = df_test[test_columns[3:130]].min(axis=1)



df_train['all_feature_max'] = df_train[train_columns[4:131]].max(axis=1)

df_test['all_feature_max']  = df_test[test_columns[3:130]].max(axis=1)



df_train['all_feature_std'] = df_train[train_columns[4:131]].std(axis=1)

df_test['all_feature_std']  = df_test[test_columns[3:130]].std(axis=1)
df_train.drop('end_time_seconds_youtube_clip',axis=1,inplace=True)

df_test.drop('end_time_seconds_youtube_clip',axis=1,inplace=True)
df_train.head()
df_test.head()
df_train_concat = df_train.groupby('vid_id').mean()

df_test_concat  = df_test.groupby('vid_id').mean()
df_train_concat.head()
X = df_train_concat.drop(['is_turkey'],axis=1)

y = df_train_concat['is_turkey']
def cross_validation(train_set, target, test_set, nfold, cv_type, seed, shuf, model):

    if cv_type == "KFold":

        kf = KFold(n_splits=nfold, random_state=seed, shuffle=shuf)

        split = kf.split(train_set)

    else:

        kf = StratifiedKFold(n_splits=nfold, shuffle=shuf, random_state=seed)

        split = kf.split(train_set, target)

    

    oof_preds = np.zeros(train_set.shape[0])

    oof_test = np.zeros(test_set.shape[0])

    for i, (train_index, val_index) in enumerate(split):

        x_tr, x_val = train_set.iloc[train_index], train_set.iloc[val_index]

        y_tr, y_val = target[train_index], target[val_index]

        params = {'random_state':seed}

        model.set_params(**params)

        model.fit(x_tr, y_tr)

        oof_preds[val_index] = model.predict_proba(x_val)[:,1]

        oof_test += model.predict_proba(test_set)[:,1] / kf.n_splits

        print("Fold %s ROC:" %str(i+1), np.round(roc_auc_score(y_val, (oof_preds[val_index])),5))

    

    print("CV Score:", np.round(roc_auc_score(target, oof_preds),5))

    return oof_preds.reshape(-1,1), oof_test.reshape(-1,1)
from sklearn.ensemble import RandomForestClassifier



RFC = RandomForestClassifier(n_estimators=160, min_samples_split=3)

oof_train_rf, oof_test_rf = cross_validation(train_set=X, target=y, test_set=df_test_concat, cv_type="SKFold", nfold=5, seed=2018, shuf=True, model=RFC)
fpr, tpr, thresholds = roc_curve(y, oof_train_rf)

roc_auc = auc(fpr, tpr)



sns.set('talk', 'whitegrid', 'dark', font_scale=1.2,rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

lw = 2

plt.figure()

plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([-0.1, 1.1])

plt.ylim([-0.1, 1.1])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Random Forest ROC')

plt.legend(loc="lower right")

plt.show()
y_pred_proba_RFC = oof_test_rf

y_pred_proba_RFC = pd.DataFrame(y_pred_proba_RFC,columns=['is_turkey'])

df_test_concat.reset_index(inplace=True)

df_test_concat['is_turkey'] = y_pred_proba_RFC['is_turkey']

df_sub = df_test_concat[['vid_id','is_turkey']] 

df_test_concat.drop('is_turkey',axis=1,inplace=True)



df_final_RFC = pd.merge(sample,df_sub,on='vid_id')

df_final_RFC.drop('is_turkey_x',axis=1,inplace=True)

df_final_RFC.columns = ['vid_id', 'is_turkey']



df_final_RFC.to_csv('submission_RFC.csv',index=False)

df_test_concat.set_index('vid_id',inplace=True)
from xgboost import XGBClassifier



XGB = XGBClassifier(max_depth=3, learning_rate=0.07, n_estimators=110, n_jobs=4)

oof_train_xgb, oof_test_xgb = cross_validation(train_set=X, target=y, test_set=df_test_concat, cv_type="SKFold", nfold=5, seed=2018, shuf=True, model=XGB)
fpr, tpr, thresholds = roc_curve(y, oof_train_xgb)

roc_auc = auc(fpr, tpr)



sns.set('talk', 'whitegrid', 'dark', font_scale=1.2,rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

lw = 2

plt.figure()

plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([-0.1, 1.1])

plt.ylim([-0.1, 1.1])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('XGBOOST ROC')

plt.legend(loc="lower right")

plt.show()
y_pred_proba_XGB = oof_test_xgb 

y_pred_proba_XGB = pd.DataFrame(y_pred_proba_XGB,columns=['is_turkey'])



df_test_concat.reset_index(inplace=True)

df_test_concat['is_turkey'] = y_pred_proba_XGB['is_turkey']

df_sub = df_test_concat[['vid_id','is_turkey']] 

df_test_concat.drop('is_turkey',axis=1,inplace=True)



df_final_XGB = pd.merge(sample,df_sub,on='vid_id')

df_final_XGB.drop('is_turkey_x',axis=1,inplace=True)

df_final_XGB.columns = ['vid_id', 'is_turkey']



df_final_XGB.to_csv('submission_XGB.csv',index=False)

df_test_concat.set_index('vid_id',inplace=True)
from lightgbm import LGBMClassifier



LGBC = LGBMClassifier(max_depth=-1, n_estimators=75, num_leaves=31)

oof_train_lgb, oof_test_lgb = cross_validation(train_set=X, target=y, test_set=df_test_concat, cv_type="SKFold", nfold=5, seed=2018, shuf=True, model=LGBC)
fpr, tpr, thresholds = roc_curve(y,oof_train_lgb)

roc_auc = auc(fpr, tpr)



sns.set('talk', 'whitegrid', 'dark', font_scale=1.2,rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

lw = 2

plt.figure()

plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([-0.1, 1.1])

plt.ylim([-0.1, 1.1])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Light GBM ROC')

plt.legend(loc="lower right")

plt.show()
y_pred_proba_LGBC = oof_test_lgb

y_pred_proba_LGBC = pd.DataFrame(y_pred_proba_LGBC,columns=['is_turkey'])

df_test_concat.reset_index(inplace=True)

df_test_concat['is_turkey'] = y_pred_proba_LGBC['is_turkey']

df_sub = df_test_concat[['vid_id','is_turkey']] 

df_test_concat.drop('is_turkey',axis=1,inplace=True)



df_final_LGBC = pd.merge(sample,df_sub,on='vid_id')

df_final_LGBC.drop('is_turkey_x',axis=1,inplace=True)

df_final_LGBC.columns = ['vid_id', 'is_turkey']



df_final_LGBC.to_csv('submission_LGBM.csv',index=False)

df_test_concat.set_index('vid_id',inplace=True)
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(C=0.00001,penalty='l2', solver="sag", max_iter=100)

oof_train_lr, oof_test_lr = cross_validation(train_set=X, target=y, test_set=df_test_concat, cv_type="SKFold", nfold=5, seed=2018, shuf=True, model=LR)
fpr, tpr, thresholds = roc_curve(y,oof_train_lr)

roc_auc = auc(fpr, tpr)



sns.set('talk', 'whitegrid', 'dark', font_scale=1.2,rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

lw = 2

plt.figure()

plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (AUC = %0.4f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([-0.1, 1.1])

plt.ylim([-0.1, 1.1])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Logistic Regression ROC')

plt.legend(loc="lower right")

plt.show()
y_pred_proba_LR =  oof_test_lr

y_pred_proba_LR = pd.DataFrame(y_pred_proba_LR,columns=['is_turkey'])

df_test_concat.reset_index(inplace=True)

df_test_concat['is_turkey'] = y_pred_proba_LR['is_turkey']

df_sub = df_test_concat[['vid_id','is_turkey']] 

df_test_concat.drop('is_turkey',axis=1,inplace=True)



df_final_LR = pd.merge(sample,df_sub,on='vid_id')

df_final_LR.drop('is_turkey_x',axis=1,inplace=True)

df_final_LR.columns = ['vid_id', 'is_turkey']



df_final_LR.to_csv('submission_LR.csv',index=False)

df_test_concat.set_index('vid_id',inplace=True)
df_corr = pd.DataFrame()

df_corr['LGBC'] = df_final_LGBC['is_turkey']

df_corr['XGB']  = df_final_XGB['is_turkey']

df_corr['RFC']  = df_final_RFC['is_turkey']

df_corr['LR']  = df_final_LR['is_turkey']

df_corr.corr()
W_RFC = 0.05

W_XGB = 0.1

W_LGB = 0.4

W_LR  = 0.45



fpr_rfc, tpr_rfc, thresholds_rfc = roc_curve(y, oof_train_rf)

roc_auc_rfc = auc(fpr_rfc, tpr_rfc)



fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y, oof_train_xgb)

roc_auc_xgb = auc(fpr_xgb, tpr_xgb)



fpr_lgb, tpr_lgb, thresholds_lgb = roc_curve(y, oof_train_lgb)

roc_auc_lgb = auc(fpr_lgb, tpr_lgb)



fpr_lr, tpr_lr, thresholds_lr = roc_curve(y, oof_train_lr)

roc_auc_lr = auc(fpr_lr, tpr_lr)



final_roc_score = (W_RFC * auc(fpr_rfc, tpr_rfc) 

                 + W_XGB * auc(fpr_xgb, tpr_xgb) 

                 + W_LGB * auc(fpr_lgb, tpr_lgb) 

                 + W_LR  * auc(fpr_lr, tpr_lr))



print('Random Forest Score: ',roc_auc_rfc)

print('XG Boost Score:      ',roc_auc_xgb)

print('Light GBM:           ',roc_auc_lgb)

print('Logistic Regression: ',roc_auc_lr)

print('Blend Score:         ',final_roc_score)
df_final_blend = df_final_LGBC



df_final_blend['is_turkey'] = W_LGB * df_final_LGBC['is_turkey'] + W_XGB * df_final_XGB['is_turkey'] + W_RFC * df_final_RFC['is_turkey'] + W_LR * df_final_LR['is_turkey']

        

df_final_blend.to_csv('submission_blend.csv',index=False)
train_stack = np.round(np.concatenate((oof_train_lr, oof_train_rf, oof_train_lgb, oof_train_xgb), axis=1))

test_stack = np.round(np.concatenate((oof_test_lr, oof_test_rf, oof_test_lgb, oof_test_xgb), axis=1))
lr = LogisticRegression(C=0.1)

lr.fit(train_stack, y)

stack_preds = lr.predict_proba(test_stack)[:,1:]
y_pred_proba_stacking =  stack_preds

y_pred_proba_stacking = pd.DataFrame(y_pred_proba_stacking,columns=['is_turkey'])

df_test_concat.reset_index(inplace=True)

df_test_concat['is_turkey'] = y_pred_proba_stacking['is_turkey']

df_sub = df_test_concat[['vid_id','is_turkey']] 

df_test_concat.drop('is_turkey',axis=1,inplace=True)



df_final_stacking = pd.merge(sample,df_sub,on='vid_id')

df_final_stacking.drop('is_turkey_x',axis=1,inplace=True)

df_final_stacking.columns = ['vid_id', 'is_turkey']



df_final_stacking.to_csv('submission_stacking.csv',index=False)

df_test_concat.set_index('vid_id',inplace=True)