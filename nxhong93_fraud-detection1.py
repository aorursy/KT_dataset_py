!pip install pyod
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

import os

import gc

import math

from datetime import datetime



import xgboost

from xgboost import XGBClassifier

import lightgbm as lgb_

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier



from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

from sklearn.cluster import DBSCAN

from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler



from pyod.models.hbos import HBOS

from pyod.models.iforest import IForest

from pyod.models.cblof import CBLOF



from imblearn.under_sampling import TomekLinks, RandomUnderSampler, ClusterCentroids

from imblearn.over_sampling import SMOTE

from imblearn.combine import SMOTETomek



sns.set_style("darkgrid")
file_path = '../input/paysim1/PS_20174392719_1491204439457_log.csv'

fold_split = 5



kf = KFold(n_splits=fold_split, random_state=37, shuffle=True)
df = pd.read_csv(file_path)

df.tail()
df.info()
plt.figure(figsize=(14, 10))



sns.countplot(data=df, y='type', order=df['type'].value_counts().index)

plt.show()
fraud_type = df.groupby('type')['isFraud'].value_counts().reset_index(name='Number of trans')

fraud_type
trans_each_desh = df['nameDest'].value_counts().reset_index(name='Number of trans')

trans_each_desh = trans_each_desh[trans_each_desh['Number of trans'] > 1]



trans_each_desh
acc_fraud = df[df['isFraud']==1]['nameOrig']

dest_fraud = df[df['isFraud']==1]['nameDest'].unique()



dest_fraud
fraud_in_desh = trans_each_desh[trans_each_desh['index'].isin(dest_fraud)].reset_index(drop=True)

fraud_in_desh
df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]

df.reset_index(drop=True, inplace=True)



feature_name = ['type_' + str(i) for i in df['type'].unique()]

type_ = pd.get_dummies(df['type'])

type_.columns = feature_name



df = pd.concat([df, type_], 1)

df.head()
train_df, val_df = train_test_split(df, test_size=0.25, random_state=37, stratify=df['isFraud'])

train_df.reset_index(drop=True, inplace=True)

val_df.reset_index(drop=True, inplace=True)



try:

    del df

except:

    print('df not exist!')



gc.collect()    
%%time



def process_data(df):

    

    df['errorBalanceOrg'] = df['amount'] + df['newbalanceOrig'] - df['oldbalanceOrg']

    df['errorBalanceDest'] = df['amount'] + df['newbalanceDest'] - df['oldbalanceDest']

    

    return df



val_df = process_data(val_df)

train_df = process_data(train_df)



del train_df['step'], train_df['type'], train_df['nameOrig'], train_df['nameDest'], train_df['isFlaggedFraud']

del val_df['step'], val_df['type'], val_df['nameOrig'], val_df['nameDest'], val_df['isFlaggedFraud']



gc.collect()



print(f'train size: {train_df.shape}, validation size: {val_df.shape}')

val_df.head()
column_train = [i for i in train_df.columns if i!='isFraud']



print(column_train)
columns_normalize = [i for i in column_train if isinstance(train_df.loc[0, i], np.float64)]



def normalize(col, type_='StandardScaler'):

    

    if type_ == 'StandardScaler':

        clf = StandardScaler()

    elif type_ == 'MinMaxScaler':

        clf = MinMaxScaler()

    elif type_ == 'RobustScaler':

        clf = RobustScaler()

    else:

        print('Error type in!')

    

    train_df[col] = clf.fit_transform(train_df[col])

    val_df[col] = clf.transform(val_df[col])

    

    return train_df, val_df

    

train_df, val_df = normalize(columns_normalize)

val_df.head()
def scatter_draw(data, feature1, feature2, feature3, fraud_feature):

    

    data_normal = data[fraud_feature==0]

    data_fraud = data[fraud_feature==1]

    

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    

    ax.scatter(data_normal[feature1], data_normal[feature2], data_normal[feature3], marker='o', c='blue', label='normal')

    ax.scatter(data_fraud[feature1], data_fraud[feature2], data_fraud[feature3], marker='o', c='red', label='fraud')

    

    ax.set_xlabel(feature1)

    ax.set_ylabel(feature2)

    ax.set_zlabel(feature3)

    

    plt.legend(loc='upper left')

    

    plt.show()

    





def roc_line(preds, titles):

    

    plt.figure(figsize=(10, 10))

    label = val_df.loc[:, 'isFraud']

    for pred, title in zip(preds, titles):

        fpr, tpr, threshold = roc_curve(label, pred)

        auc_score = auc(fpr, tpr)

        

        plt.plot(fpr, tpr, label=f'{title}: {auc_score:.2f}')

        

    plt.plot([0, 1], [0, 1], linestyle='--', color='blue')

    plt.legend(loc='lower right')

    plt.xlabel('False Positive Rate ')

    plt.ylabel('True Positive Rate')

    

    plt.show()

    

    

    

def heatmap_cal(pred):

    

    confusion = confusion_matrix(val_df['isFraud'].values, pred)

    print(classification_report(val_df['isFraud'].values, pred))

    print(55*'-')

    sns.heatmap(confusion, annot=True, annot_kws={"size": 16}, fmt='.0f')

    plt.xlabel('Predict label')

    plt.ylabel('True label')

    plt.show()

    

    

def feature_important(importances):

    

    data = pd.DataFrame({'feature': column_train, 'important': importances})

    

    plt.figure(figsize=(15, 15))

    plt.title('Feature Importances')

    sns.barplot(data=data.sort_values('important', ascending=False), x='important', y='feature')

    plt.xlabel('Relative Importance')

    plt.show()

    

    return data    
scatter_draw(val_df, 'amount', 'oldbalanceOrg', 'oldbalanceDest', val_df['isFraud'].values)
fraud_ratio = np.count_nonzero(train_df['isFraud'])/len(train_df)

print(fraud_ratio)
%%time



isolation = IForest(contamination=fraud_ratio)

isolation.fit(train_df.loc[:, column_train])



pred_isolation = isolation.predict(val_df.loc[:, column_train])

pred_prob_isolation = isolation.predict_proba(val_df.loc[:, column_train])

print(f'Number of abnormal detection: {np.count_nonzero(pred_isolation)}')
heatmap_cal(pred_isolation)
scatter_draw(val_df, 'amount', 'oldbalanceOrg', 'oldbalanceDest', pred_isolation)
roc_line([pred_prob_isolation[:, 1]], ['isolation'])
clf = LogisticRegression(penalty='none', solver='newton-cg')

clf.fit(train_df.loc[:, column_train], train_df.loc[:, ['isFraud']].values.flatten())



pred_log = clf.predict(val_df.loc[:, column_train])

pred_prob_log = clf.predict_proba(val_df.loc[:, column_train])

print(f'Number of abnormal detection: {sum(pred_log)}')
heatmap_cal(pred_log)
scatter_draw(val_df, 'amount', 'oldbalanceOrg', 'oldbalanceDest', pred_log)
smote = SMOTE(random_state=0)

train_smote, label_smote = smote.fit_sample(train_df.loc[:, column_train], train_df.loc[:, 'isFraud'])
clf.fit(train_smote, label_smote.values.flatten())



pred_log_smote = clf.predict(val_df.loc[:, column_train])

pred_prob_log_smote = clf.predict_proba(val_df.loc[:, column_train])

print(f'Number of abnormal detection: {np.count_nonzero(pred_log_smote)}')
heatmap_cal(pred_log_smote)
scatter_draw(val_df, 'amount', 'oldbalanceOrg', 'oldbalanceDest', pred_log_smote)
random = RandomUnderSampler()

train_random, label_random = random.fit_sample(train_df.loc[:, column_train], train_df.loc[:, 'isFraud'])
clf.fit(train_random, label_random.values.flatten())



pred_log_random = clf.predict(val_df.loc[:, column_train])

pred_prob_log_random = clf.predict_proba(val_df.loc[:, column_train])

print(f'Number of abnormal detection: {np.count_nonzero(pred_log_random)}')
heatmap_cal(pred_log_random)
scatter_draw(val_df, 'amount', 'oldbalanceOrg', 'oldbalanceDest', pred_log_random)
roc_line([pred_prob_log, pred_prob_log_smote, pred_prob_log_random], ['logistic', 'logistic_smote', 'logistic_random'])
%%time



params = {'n_estimators': [100],

          'max_depth': [25],

          'criterion': ['gini']}



rdf = RandomForestClassifier(min_samples_leaf=50)



grid_forest = GridSearchCV(rdf, param_grid = params, cv=5, n_jobs=1, scoring='roc_auc', verbose=3)

grid_forest.fit(train_df.loc[:, column_train], train_df.loc[:, 'isFraud'].values.flatten())



grid_forest.best_params_
rdf_importances = grid_forest.best_estimator_.feature_importances_

data = feature_important(rdf_importances)
rdf_best = grid_forest.best_estimator_

pred_forest = rdf_best.predict(val_df.loc[:, column_train])

pred_prob_forest = rdf_best.predict_proba(val_df.loc[:, column_train])

print(f'Number of abnormal detection: {np.count_nonzero(pred_forest)}')
heatmap_cal(pred_forest)
scatter_draw(val_df, 'amount', 'oldbalanceOrg', 'oldbalanceDest', pred_forest)
%%time



pred_prob_xgb = np.zeros([len(val_df), fold_split])

xgb_importance = np.zeros([len(column_train), fold_split])



for idx, (train_index, val_index) in enumerate(kf.split(train_df)):

    xgb = XGBClassifier(n_estimators=100, max_depth=25, criterion='gini', learning_rate=0.02, eval_metric='auc')

    

    print(f'Fold_{idx}:')

    train_ = train_df.iloc[train_index]

    val_ = train_df.iloc[val_index]

    

    train_.reset_index(inplace=True, drop=True)

    val_.reset_index(inplace=True, drop=True)

    

        

    xgb.fit(train_.loc[:, column_train], train_.loc[:, 'isFraud'].values.flatten(), 

            eval_set=[(val_.loc[:, column_train], val_.loc[:, 'isFraud'].values.flatten())],

            verbose=10, early_stopping_rounds=30)

    

    pred_prob_xgb[:, idx] = xgb.predict_proba(val_df.loc[:, column_train], ntree_limit=xgb.best_ntree_limit)[:, 1]

    xgb_importance[:, idx] = xgb.feature_importances_

    

    del xgb

    

    gc.collect()

    

pred_prob_xgb = np.mean(pred_prob_xgb, axis=1)

pred_xgb = np.where(pred_prob_xgb >= 0.5, 1, 0)

xgb_importance = np.mean(xgb_importance, axis=1)
data = feature_important(list(xgb_importance))
heatmap_cal(pred_xgb)
scatter_draw(val_df, 'amount', 'oldbalanceOrg', 'oldbalanceDest', pred_xgb)
%%time



pred_prob_lgb = np.zeros([len(val_df), fold_split])

lgb_importance = np.zeros([len(column_train), fold_split])



for idx, (train_index, val_index) in enumerate(kf.split(train_df)):

    lgb = LGBMClassifier(n_estimators=1000, max_depth=25,

                         

                         learning_rate=0.01, eval_metric='roc_auc', 

                         objective='binary', boosting='gbdt')

    

    print(f'Fold_{idx}:')

    train_ = train_df.iloc[train_index]

    val_ = train_df.iloc[val_index]

    

    train_.reset_index(inplace=True, drop=True)

    val_.reset_index(inplace=True, drop=True)

    

    lgb.fit(train_.loc[:, column_train], train_.loc[:, 'isFraud'].values.flatten(), 

            eval_set=[(val_.loc[:, column_train], val_.loc[:, 'isFraud'].values.flatten())],

            eval_metric=['auc'], 

            verbose=100, early_stopping_rounds=500)

    

    pred_prob_lgb[:, idx] = lgb.predict_proba(val_df.loc[:, column_train])[:, 1]

    lgb_importance[:, idx] = lgb.feature_importances_

    

    del lgb

    

    gc.collect()

    

    

pred_prob_lgb = np.mean(pred_prob_lgb, axis=1)

pred_lgb = np.where(pred_prob_lgb >= 0.5, 1, 0)

lgb_importance = np.mean(lgb_importance, axis=1)    
data = feature_important(list(lgb_importance))
heatmap_cal(pred_lgb)
val_add_label_df = val_df.copy()

val_add_label_df['isFraud'] = pred_lgb



train_add_label_df = pd.concat([train_df, val_add_label_df])

train_add_label_df.reset_index(inplace=True, drop=True)



train_add_label_df.tail()
%%time



pred_prob_lgb = np.zeros([len(val_df), fold_split])

lgb_importance = np.zeros([len(column_train), fold_split])



for idx, (train_index, val_index) in enumerate(kf.split(train_add_label_df)):

    lgb = LGBMClassifier(n_estimators=1000, max_depth=25, 

                         learning_rate=0.01, eval_metric='auc', 

                         objective='binary', boosting='gbdt')

    

    print(f'Fold_{idx}:')

    train_ = train_add_label_df.iloc[train_index]

    val_ = train_add_label_df.iloc[val_index]

    

    train_.reset_index(inplace=True, drop=True)

    val_.reset_index(inplace=True, drop=True)

    

    lgb.fit(train_.loc[:, column_train], train_.loc[:, 'isFraud'].values.flatten(), 

            eval_set=[(val_.loc[:, column_train], val_.loc[:, 'isFraud'].values.flatten())],

            eval_metric=['auc'], 

            verbose=100, early_stopping_rounds=500)

    

    pred_prob_lgb[:, idx] = lgb.predict_proba(val_df.loc[:, column_train])[:, 1]

    lgb_importance[:, idx] = lgb.feature_importances_

    

    del lgb

    

    gc.collect()

    

    

pred_prob_lgb = np.mean(pred_prob_lgb, axis=1)

pred_lgb = np.where(pred_prob_lgb >= 0.5, 1, 0)

lgb_importance = np.mean(lgb_importance, axis=1)    
heatmap_cal(pred_lgb)
scatter_draw(val_df, 'amount', 'oldbalanceOrg', 'oldbalanceDest', pred_lgb)
%%time



pred_prob_catboost = np.zeros([len(val_df), fold_split])

catboost_importance = np.zeros([len(column_train), fold_split])



for idx, (train_index, val_index) in enumerate(kf.split(train_df)):

    catboost = CatBoostClassifier(iterations=500, depth=8, l2_leaf_reg=3, loss_function='CrossEntropy')

    

    print(f'Fold_{idx}:')

    train_ = train_df.iloc[train_index]

    val_ = train_df.iloc[val_index]

    

    train_.reset_index(inplace=True, drop=True)

    val_.reset_index(inplace=True, drop=True)

    

        

    catboost.fit(train_.loc[:, column_train], train_.loc[:, 'isFraud'].values.flatten(), 

                eval_set=[(val_.loc[:, column_train], val_.loc[:, 'isFraud'].values.flatten())],

                verbose=100, early_stopping_rounds=30)

    

    pred_prob_catboost[:, idx] = catboost.predict_proba(val_df.loc[:, column_train])[:, 1]

    catboost_importance[:, idx] = catboost.feature_importances_

    

    del catboost

    

    gc.collect()

    

pred_prob_catboost = np.mean(pred_prob_catboost, axis=1)

pred_catboost = np.where(pred_prob_catboost >= 0.5, 1, 0)

catboost_importance = np.mean(catboost_importance, axis=1)
data = feature_important(catboost_importance)
heatmap_cal(pred_catboost)
params_catboost = {'iterations': [500],

                   'depth': [4, 6, 8],

                   'loss_function': ['Logloss', 'CrossEntropy'], 

                   'l2_leaf_reg': [3]}
scatter_draw(val_df, 'amount', 'oldbalanceOrg', 'oldbalanceDest', pred_catboost)
df_pred = pd.DataFrame({'logistic': pred_log, 'random_forest': pred_forest, 'xgb': pred_xgb, 'lgb': pred_lgb, 'catboost': pred_catboost})

df_pred['pred'] = df_pred.index.map(lambda x: 1 if (int(df_pred.loc[x, 'logistic']) + int(df_pred.loc[x, 'random_forest']) + int(df_pred.loc[x, 'xgb']) + int(df_pred.loc[x, 'lgb']) + int(df_pred.loc[x, 'catboost'])) > 2 else 0)



heatmap_cal(df_pred['pred'].values)
scatter_draw(val_df, 'amount', 'oldbalanceOrg', 'oldbalanceDest', df_pred['pred'].values)
roc_line([pred_prob_log[:, 1], pred_prob_forest[:, 1], pred_prob_xgb, pred_prob_lgb, pred_prob_catboost], 

         ['logistic', 'random_forest', 'xgb', 'lgb', 'catboost'])