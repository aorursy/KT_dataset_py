# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt
%matplotlib inline
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import lightgbm as lgb
import gc
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")
data = pd.read_csv("../input/application_train.csv")
test = pd.read_csv('../input/application_test.csv')
prev = pd.read_csv('../input/previous_application.csv')
buro = pd.read_csv('../input/bureau.csv')
buro_balance = pd.read_csv('../input/bureau_balance.csv')
credit_card  = pd.read_csv('../input/credit_card_balance.csv')
POS_CASH  = pd.read_csv('../input/POS_CASH_balance.csv')
payments = pd.read_csv('../input/installments_payments.csv')
lgbm_submission = pd.read_csv('../input/sample_submission.csv')
buro_balance.head()
prev.head()
credit_card.head()
payments.head()
y = data["TARGET"]
del data["TARGET"]
categorical_feature = [col for col in data.columns if data[col].dtype == 'object']
one_hot_df = pd.concat([data, test])
one_hot_df = pd.get_dummies(one_hot_df, columns=categorical_feature)
data = one_hot_df.iloc[:data.shape[0],:]
test = one_hot_df.iloc[data.shape[0]:,]
buro_grouped_size = buro_balance.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].size()
buro_grouped_max = buro_balance.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].max()
buro_grouped_min = buro_balance.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].min()
buro_counts = buro_balance.groupby("SK_ID_BUREAU")["STATUS"].value_counts(normalize=False)
buro_counts_unstacked = buro_counts.unstack("STATUS")

buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5','STATUS_C','STATUS_X',]
buro_counts_unstacked["MONTHS_COUNT"] = buro_grouped_size
buro_counts_unstacked["MONTHS_MIN"] = buro_grouped_min
buro_counts_unstacked["MONTHS_MAX"] = buro_grouped_max
buro = buro.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')
prev.head(10)
prev_cat_features = [pcol for pcol in prev.columns if prev[pcol].dtype == "object"]
prev = pd.get_dummies(prev, columns=prev_cat_features)
# prev.head(10)
avg_prev = prev.groupby("SK_ID_CURR").mean()
# avg_prev.head(10)
cnt_prev = prev[["SK_ID_CURR", "SK_ID_PREV"]].groupby("SK_ID_CURR").count()
avg_prev["nb_app"] = cnt_prev["SK_ID_PREV"]
del avg_prev["SK_ID_PREV"]
buro_cat_features = [bcol for bcol in buro.columns if buro[bcol].dtype == 'object']
buro = pd.get_dummies(buro, columns=buro_cat_features)
avg_buro = buro.groupby('SK_ID_CURR').mean()
avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
del avg_buro['SK_ID_BUREAU']
avg_buro.head(10)
le = LabelEncoder()
POS_CASH['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
nunique_status2.head(10)
POS_CASH['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
POS_CASH['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)
le =LabelEncoder()
credit_card['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
credit_card['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

avg_payments = payments.groupby("SK_ID_CURR").mean()
avg_payments2 = payments.groupby("SK_ID_CURR").max()
avg_payments3 = payments.groupby("SK_ID_CURR").min()
del avg_payments["SK_ID_PREV"]
data = data.merge(right=avg_prev.reset_index(), how="left", on="SK_ID_CURR")
test = test.merge(right=avg_prev.reset_index(), how="left", on="SK_ID_CURR")

data.head(10)
data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
data = data.merge(right=POS_CASH.groupby("SK_ID_CURR").mean().reset_index(), how="left", on="SK_ID_CURR")
test = test.merge(right=POS_CASH.groupby("SK_ID_CURR").mean().reset_index(), how="left", on="SK_ID_CURR")
data = data.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
data = data.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
data.shape
test = test[test.columns[data.isnull().mean() < 0.85]]
data = data[data.columns[data.isnull().mean() < 0.85]]
excluded_feats = ["SK_ID_CURR"]
features = [f for f in data.columns if f not in excluded_feats]
# bayes_cv_tuner = BayesSearchCV(
#     estimator = lgb.LGBMClassifier(
#         boosting='dart',
#         application='binary',
#         metric='auc',
#         drop_rate=0.2,
#         n_jobs=1,
#         verbose=0
#     ),
#     search_spaces = {
#         'learning_rate': (0.01, 0.3, 'uniform'),
#         'num_leaves': (1, 225),      
#         'max_depth': (0, 8),
#         'feature_fraction':(0.5, 1.0, 'uniform'),
#         "min_data_in_leaf":(20, 100),
# #         'min_child_samples': (0, 50),
#         'max_bin': (100, 1000),
#         'reg_lambda': (1e-9, 1.0, 'log-uniform'),
#         'reg_alpha': (1e-9, 1.0, 'log-uniform'),
#         'scale_pos_weight': (1,12, 'uniform'),
# },    
#     scoring = 'roc_auc',
#     cv = StratifiedKFold(
#         n_splits=3,
#         shuffle=True,
#         random_state=42
#     ),
#     n_jobs = 1,
#     n_iter = 15,   
#     verbose = 0,
#     refit = True,
#     random_state = 42
# )

# # Fit the model
# result = bayes_cv_tuner.fit(data, y, callback=status_print)
# # model = lgb.LGBMClassifier(lgbm_params)
# Best ROC-AUC: 0.7618
# Best params: {'max_bin': 783, 'max_depth': 7, 'min_child_samples': 37, 'min_child_weight': 7, 'n_estimators': 94, 'num_leaves': 92, 'reg_alpha': 0.6654390259962506, 'reg_lambda': 8.076151891962533e-06, 'scale_pos_weight': 7.642490251593845, 'subsample': 0.25371759984574854, 'subsample_freq': 9}
# Model #10
# Best ROC-AUC: 0.7711
# Best params: {'learning_rate': 0.685534641629431, 'max_bin': 112, 'max_depth': 38, 'min_child_samples': 42, 'min_child_weight': 3, 'n_estimators': 60, 'num_leaves': 25, 'reg_alpha': 1.462442068214992e-06, 'reg_lambda': 3.5571385509488406e-07, 'scale_pos_weight': 0.0052366805641386495, 'subsample': 0.7074795557274224, 'subsample_freq': 10}
data.head(10)
lgbm_params = {
    "boosting":"dart",
    "application":"binary",
    "learning_rate": 0.1,
    'reg_alpha':0.01,
    'reg_lambda': 0.01,
    "n_estimators":10000,
    "max_depth":7,
    "num_leaves":100,
    "max_bin":225,
    "drop_rate":0.02
}
model = lgb.LGBMClassifier(application="binary", boosting_type=lgbm_params["boosting"],
                          learning_rate=lgbm_params["learning_rate"],n_estimators=lgbm_params["n_estimators"],drop_rate=lgbm_params["drop_rate"],
                          num_leaves=lgbm_params["num_leaves"], max_depth=lgbm_params["max_depth"],
                          max_bin=lgbm_params["max_bin"])
#                           min_data_in_leaf=lgbm_params["min_data_in_leaf"],
#                            feature_fraction=lgbm_params["feature_fraction"],
                            
# clf = lgb.train(train_set=lgbm_train,
#                  params=lgbm_params,
#                  num_boost_round=optimum_boost_rounds)
# model = lgb.LGBMClassifier(lgbm_params)
feature_importances = np.zeros(data.shape[1])
for i in range(2):
    
    train_data, test_data, train_y, test_y = train_test_split(data, y, test_size=0.2, random_state=i)
    
    model.fit(train_data, train_y, early_stopping_rounds=100, eval_set=[(test_data, test_y)], eval_metric='auc', verbose=200)
    
    feature_importances += model.feature_importances_
feature_importances = feature_importances/2
feature_importances = pd.DataFrame({'feature': list(data.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)

feature_importances.head()
zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
print('There are %d features with 0.0 importance' % len(zero_features))
feature_importances.tail()
def plot_feature_importances(df, threshold=0.98):
    
    plt.rcParams["font.size"] = 18
    
    df = df.sort_values('importance', ascending=False).reset_index()
    
    df['importance_normalized'] = df['importance']/df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.figure(figsize=(10,6))
    ax = plt.subplot()
    
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align='center', edgecolor='k')
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    plt.figure(figsize=(8,6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of features');plt.ylabel('Cumulative Importances');
    
    plt.title("Cumulative Feature Importance");
    plt.show()
    
    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))
    
    return df
norm_feature_importances = plot_feature_importances(feature_importances)
threshold = 0.98

features_to_keep = list(norm_feature_importances[norm_feature_importances['cumulative_importance'] < threshold]['feature'])

data = data[features_to_keep]
test = test[features_to_keep]
# data = data.drop(columns = zero_features)
# test = test.drop(columns = zero_features)
# # 
print('Training shape: ', data.shape)
print('Testing shape: ', test.shape)
n_folds = 5
k_fold = KFold(n_splits=n_folds, shuffle=False, random_state=50)

feature_importances_values = np.zeros(data.shape[1])

test_predictions = np.zeros(test.shape[0])
out_of_fold = np.zeros(data.shape[0])

valid_scores = []
train_scores = []

for train_indices, test_indices in k_fold.split(data):
    
    train_data, train_y = data.iloc[train_indices], y.iloc[train_indices]
    test_data, test_y = data.iloc[test_indices], y.iloc[test_indices]
    
    model = lgb.LGBMClassifier(application="binary", boosting_type=lgbm_params["boosting"],
                          learning_rate=lgbm_params["learning_rate"],  n_estimators=lgbm_params["n_estimators"],
                          num_leaves=lgbm_params["num_leaves"],max_depth=lgbm_params["max_depth"],
                          reg_lambda=lgbm_params['reg_lambda'],reg_alpha=lgbm_params["reg_alpha"],
                          drop_rate=lgbm_params["drop_rate"], random_state=50)
    
    model.fit(train_data, train_y, eval_metric='auc', eval_set=[(test_data, test_y), (train_data, train_y)],
              eval_names=['valid', 'train'], early_stopping_rounds=100, verbose=200)
    
    best_iteration = model.best_iteration_
    feature_importances_values += model.feature_importances_ / k_fold.n_splits
    
    test_predictions += model.predict_proba(test, num_iteration=best_iteration)[:,1]/k_fold.n_splits
    
    out_of_fold[test_indices] = model.predict_proba(test_data, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
    valid_score = model.best_score_['valid']['auc']
    train_score = model.best_score_['train']['auc']
        
    valid_scores.append(valid_score)
    train_scores.append(train_score)
        
    gc.enable()
    del model, train_data, test_data
    gc.collect()
    

submission = pd.DataFrame({'SK_ID_CURR': test["SK_ID_CURR"], 'TARGET': test_predictions})
submission.to_csv("submissions.csv", index=False)
# out_df = pd.DataFrame({"SK_ID_CURR":test["SK_ID_CURR"], "TARGET":y_pred})
# out_df.to_csv("submissions.csv", index=False)
