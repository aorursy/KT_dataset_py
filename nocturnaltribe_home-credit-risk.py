import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import glob
import pathlib

import csv

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
from sklearn import metrics

import sklearn.pipeline
# import lightgbm as lgb

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
%matplotlib inline
fpath = glob.glob('../input/*.csv')
ssize = 0.1
dfnames = []
for path in fpath:
    with open(path,"r") as f:
        reader = csv.reader(f,delimiter = ",")
        data = list(reader)
        row_count = len(data)
        del data
    chunksize = int( row_count * ssize )
    print(path,row_count,chunksize)
    
    file = path.split('/')[2]
    name = 'df_' + file.split('.csv')[0]
    fcomm = name + ' = pd.read_csv(pathlib.Path(path))'#.sample( chunksize )'    
    dfnames.append(name)
    exec(fcomm)

# del df_application_train
# del df_application_test
# df_application_train = pd.read_csv('../input/application_train.csv')
# df_application_test = pd.read_csv('../input/application_test.csv')
for name in dfnames:
    fcomm = 'numrow = len(' + name + '.index)'    
    exec(fcomm)
    print( name, numrow )

df_bureau.loc[df_bureau['AMT_ANNUITY'] > .8e8, 'AMT_ANNUITY'] = np.nan
df_bureau.loc[df_bureau['AMT_CREDIT_SUM'] > 3e8, 'AMT_CREDIT_SUM'] = np.nan
df_bureau.loc[df_bureau['AMT_CREDIT_SUM_DEBT'] > 1e8, 'AMT_CREDIT_SUM_DEBT'] = np.nan
df_bureau.loc[df_bureau['AMT_CREDIT_MAX_OVERDUE'] > .8e8, 'AMT_CREDIT_MAX_OVERDUE'] = np.nan
df_bureau.loc[df_bureau['DAYS_ENDDATE_FACT'] < -10000, 'DAYS_ENDDATE_FACT'] = np.nan
df_bureau.loc[(df_bureau['DAYS_CREDIT_UPDATE'] > 0) | (df_bureau['DAYS_CREDIT_UPDATE'] < -40000), 'DAYS_CREDIT_UPDATE'] = np.nan
df_bureau.loc[df_bureau['DAYS_CREDIT_ENDDATE'] < -10000, 'DAYS_CREDIT_ENDDATE'] = np.nan

df_bureau.drop(df_bureau[df_bureau['DAYS_ENDDATE_FACT'] < df_bureau['DAYS_CREDIT']].index, inplace = True)

df_previous_application.loc[df_previous_application['AMT_CREDIT'] > 6000000, 'AMT_CREDIT'] = np.nan
df_previous_application.loc[df_previous_application['SELLERPLACE_AREA'] > 3500000, 'SELLERPLACE_AREA'] = np.nan
df_previous_application[['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 
             'DAYS_LAST_DUE', 'DAYS_TERMINATION']].replace(365243, np.nan, inplace = True)

df_POS_CASH_balance.loc[df_POS_CASH_balance['CNT_INSTALMENT_FUTURE'] > 60, 'CNT_INSTALMENT_FUTURE'] = np.nan

df_installments_payments.loc[df_installments_payments['NUM_INSTALMENT_VERSION'] > 70, 'NUM_INSTALMENT_VERSION'] = np.nan
df_installments_payments.loc[df_installments_payments['DAYS_ENTRY_PAYMENT'] < -4000, 'DAYS_ENTRY_PAYMENT'] = np.nan

df_credit_card_balance.loc[df_credit_card_balance['AMT_PAYMENT_CURRENT'] > 4000000, 'AMT_PAYMENT_CURRENT'] = np.nan
df_credit_card_balance.loc[df_credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL'] > 1000000, 'AMT_CREDIT_LIMIT_ACTUAL'] = np.nan
df_application_train.head(2)
df_application_train.info()
# df_bureau1 = pd.merge( 
#              pd.merge( df_application_test,df_bureau, on='SK_ID_CURR', how='left' ),
#                        df_bureau_balance, on='SK_ID_BUREAU', how='left' )
# df_bureau_balance.set_index('SK_ID_BUREAU').head(2)
# def join_df(a,b,key,rsuff):
#     temp = a.join(b.set_index(key), on=key, rsuffix=rsuff, how='left' )
#     return temp;

# def join_df_serise(dfra, b1,b2, p1,p2,p3,p4):
#     tempb= join_df(b1,b2,'SK_ID_BUREAU','')
#     temp = join_df(
#            join_df(
#            join_df(
#            join_df(dfra,p1,'SK_ID_CURR','_p1'), 
#                         p2,'SK_ID_CURR','_p2'), 
#                         p3,'SK_ID_CURR','_p3') , 
#                         p4,'SK_ID_CURR','_p4')
#     temp = join_df(temp,tempb,'SK_ID_CURR','_b')
#     del tempb
#     return temp;
def join_df(a,b,key,rsuff):
    result = pd.merge(a, b.drop_duplicates(subset=key), on=key, 
                      left_index=True, how='left', sort=False, suffixes=('',rsuff))
    print(len(result[key]))
    return result;
def join_df_serise(dfra, b1,b2, p1,p2,p3,p4):
    
    # Remove some empty features
    dfra.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 
               'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 
               'FLAG_DOCUMENT_21'], axis = 1, inplace = True)
    
    temp = join_df(
           join_df(
           join_df(
           join_df(
           join_df(
           join_df(dfra,p1,'SK_ID_CURR','_p1'), 
                        p2, ['SK_ID_CURR','SK_ID_PREV'], '_p2'), 
                        p3, ['SK_ID_CURR','SK_ID_PREV'],'_p3') , 
                        p4, ['SK_ID_CURR','SK_ID_PREV'],'_p4') , 
                        b1,'SK_ID_CURR',''), 
                        b2,'SK_ID_BUREAU','_b')
    return temp;
# Remove some rows with values not present in test set
df_application_train.drop(df_application_train[df_application_train['CODE_GENDER'] == 'XNA'].index, inplace = True)
df_application_train.drop(df_application_train[df_application_train['NAME_INCOME_TYPE'] == 'Maternity leave'].index, inplace = True)
df_application_train.drop(df_application_train[df_application_train['NAME_FAMILY_STATUS'] == 'Unknown'].index, inplace = True)

df_train = join_df_serise ( df_application_train, 
                            df_bureau.drop('AMT_ANNUITY',axis=1),df_bureau_balance, 
                            df_previous_application,df_POS_CASH_balance,
                            df_installments_payments,df_credit_card_balance )
df_train.info()
df_test = join_df_serise( df_application_test, 
                            df_bureau.drop('AMT_ANNUITY',axis=1),df_bureau_balance, 
                            df_previous_application,df_POS_CASH_balance,
                            df_installments_payments,df_credit_card_balance )
df_train.head(2)
del df_application_test
del df_POS_CASH_balance
del df_credit_card_balance
del df_installments_payments
del df_application_train
del df_bureau
del df_previous_application
del df_bureau_balance
df_train.fillna(0, inplace=True)
df_train.info()
for col in list(df_train.columns):
    df_list = list(df_train[col].unique())
    
    df_list = [x for x in df_list if x != 0]
    x = None
    for x in df_list:
        if isinstance( x, str ):
            xmax = len(df_list) + 1
#             print(col)
#             print(len(df_list),df_list)
            df_train[col].replace( df_list, list(range(1,xmax)), inplace=True )
            break
df_train.info()
sns.pairplot(df_train[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_DOWN_PAYMENT','RATE_DOWN_PAYMENT','TARGET']].dropna(),
             hue='TARGET',aspect=1.5)
sns.lmplot(  x='AMT_DOWN_PAYMENT',y='AMT_CREDIT', hue='TARGET',
                  fit_reg=False, data=df_train )
plt.xscale('log')
plt.xlim(10**-2,10**7)
# plt.yscale('log')
# plt.ylim(10**-1,10**7)
plt.show()
sns.lmplot(  x='AMT_CREDIT_SUM_DEBT',y='AMT_CREDIT_SUM_OVERDUE', hue='TARGET',
                  fit_reg=False, data=df_train )
plt.xscale('log')
plt.xlim(0.01,10**8)
plt.yscale('log')
plt.ylim(0.01,10**8)
plt.show()
sns.lmplot(  x='AMT_DOWN_PAYMENT',y='RATE_DOWN_PAYMENT', hue='TARGET',
                  fit_reg=False, data=df_train )
plt.xscale('log')
plt.xlim(0.01,10**7)
# plt.yscale('log')
# plt.ylim(10**3,10**7)
plt.show()
X_train = df_train.drop('TARGET',axis=1)
y_train = df_train['TARGET']
# XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.30)
# import sklearn.pipeline
# scaler = sklearn.preprocessing.StandardScaler()

# import lightgbm as lgb
# # train
# gbm = lgb.LGBMRegressor(objective='binary',#'regression',
#                         metric = 'binary_logloss',
#                         boosting_type='gbdt',
#                         num_leaves=1001,
#                         learning_rate=0.0005,
#                         n_estimators=200)

# steps = [('scaler', scaler),
#         ('GBM', gbm)]

# pipeline = sklearn.pipeline.Pipeline(steps)

# ### fit pipeline on X_train and y_train
# pipeline.fit( XX_train, yy_train)

# ### call pipeline.predict() on X_test data to make a set of test predictions
# yy_gbm = pipeline.predict( XX_test )
# mean = 0.5
# results = yy_gbm + (1-mean)
# predictions  = list(map(int, results))

# print('MAE:', metrics.mean_absolute_error(yy_test, predictions))
# print('MSE:', metrics.mean_squared_error(yy_test, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(yy_test, predictions)))
df_test.fillna(0, inplace=True)
for col in list(df_test.columns):
    df_list = list(df_test[col].unique())
    
    df_list = [x for x in df_list if x != 0]
    x = None
    for x in df_list:
        if isinstance( x, str ):
            xmax = len(df_list) + 1
#             print(col)
#             print(len(df_list),df_list)
            df_test[col].replace( df_list, list(range(1,xmax)), inplace=True )
            break
X_test = df_test
print( len(X_train.columns),len(X_test.columns) )
print( len(y_train), len(X_train['SK_ID_CURR']) )
# import sklearn.pipeline
# scaler = sklearn.preprocessing.StandardScaler()

# import lightgbm as lgb
# # train
# # gbm = lgb.LGBMRegressor(objective='binary',#'regression',
# #                         metric = 'binary_logloss',
# #                         boosting_type='gbdt',
# #                         num_leaves=1001,
# #                         learning_rate=0.0005,
# #                         n_estimators=200)

# gbm = lgb.LGBMRegressor(
#             nthread=4,
#             n_estimators=50000,
#             learning_rate=0.0001,
#             num_leaves=34,
#             colsample_bytree=0.9497036,
#             subsample=0.8715623,
#             max_depth=8,
#             reg_alpha=0.041545473,
#             reg_lambda=0.0735294,
#             min_split_gain=0.0222415,
#             min_child_weight=39.3259775,
#             silent=-1,
#             verbose=-1) 

# steps = [('scaler', scaler),
#         ('GBM', gbm)]

# pipeline = sklearn.pipeline.Pipeline(steps)

# ### fit pipeline on X_train and y_train
# pipeline.fit( X_train, y_train)

# ### call pipeline.predict() on X_test data to make a set of test predictions
# y_gbm = pipeline.predict( X_test )
# sns.distplot(y_gbm)
# target = []
# for y in y_gbm:
#     target.append( '{:.{prec}f}'.format(y, prec=1) ) 
# pd.DataFrame( { 'SK_ID_CURR':list(df_test['SK_ID_CURR']),
#                 'TARGET':target } ).set_index('SK_ID_CURR').to_csv('sample_submission.csv', sep=',')
params = {
         'colsample_bytree': 0.41780363323466824,
         'learning_rate': 0.010324510220774302,
         'num_leaves': 97,
         'subsample': 0.8029241575078704,
         'max_depth': 6,
         'reg_alpha': 0.03711256722090833,
         'reg_lambda': 0.0691714496715749,
         'min_split_gain': 0.024536673831831966,
         'min_child_weight': 44.94997450884206
}
def train_model(data_, test_, y_, folds_):

    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
                
        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(**params, n_estimators = 10000, nthread = 4)
#         clf = LGBMClassifier(
#             objective='binary',
#             metric = 'auc',
#             boosting_type='gbdt',
#             nthread=4,
#             num_leaves=100,
#             learning_rate=0.03,
#             n_estimators=1000 )
        
        clf.fit(
            trn_x,
            trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric='auc',
            verbose=100,
            early_stopping_rounds=100  #30
        )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats],
            num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' %
              (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
#         gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))

    test_['TARGET'] = sub_preds

    df_oof_preds = pd.DataFrame({'SK_ID_CURR':ids, 'TARGET':y, 'PREDICTION':oof_preds})
    df_oof_preds = df_oof_preds[['SK_ID_CURR', 'TARGET', 'PREDICTION']]

    return oof_preds, df_oof_preds, test_[['SK_ID_CURR', 'TARGET']], feature_importance_df, roc_auc_score(y, oof_preds)
#https://www.kaggle.com/tilii7
def display_importances(feature_importance_df_):
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").mean().sort_values(
            by="importance", ascending=False)[:50].index

    best_features = feature_importance_df_.loc[
        feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
#https://www.kaggle.com/tilii7
def display_roc_curve(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6, 6))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    plt.plot(
        [0, 1], [0, 1],
        linestyle='--',
        lw=2,
        color='r',
        label='Luck',
        alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(
        fpr,
        tpr,
        color='b',
        label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
        lw=2,
        alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
#https://www.kaggle.com/tilii7
def display_precision_recall(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6, 6))

    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(
        precision,
        recall,
        color='b',
        label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
        lw=2,
        alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM Recall / Precision')
    plt.legend(loc="best")
    plt.tight_layout()
data = X_train
test = X_test
y    = y_train
ids  = X_train['SK_ID_CURR']

# Create Folds
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)
# Train model and get oof and test predictions
oof_preds, df_oof_preds, test_preds, importances, score = train_model(data, test, y, folds)
# Display a few graphs
display_importances(feature_importance_df_=importances)
folds_idx = [(trn_idx, val_idx)
             for trn_idx, val_idx in folds.split(data, y)]
display_roc_curve(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx)
display_precision_recall(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx)
test_preds.set_index('SK_ID_CURR').to_csv('sample_submission.csv', sep=',')
sns.distplot(test_preds['TARGET'])
