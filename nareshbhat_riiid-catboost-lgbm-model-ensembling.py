import numpy as np 

import pandas as pd 

from collections import Counter

import pandas_profiling as pp

from sklearn.model_selection import StratifiedKFold

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score

import warnings

warnings.simplefilter('ignore')
import riiideducation

env = riiideducation.make_env()
train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', low_memory=False,nrows=10**5,  dtype={'row_id': 'int64',

    'timestamp': 'int64','user_id': 'int32','content_id': 'int16', 'content_type_id': 'int8','task_container_id': 'int16',

    'user_answer': 'int8','answered_correctly': 'int8','prior_question_elapsed_time': 'float32', 'prior_question_had_explanation': 'boolean'} )    

train.head()
pp_train = pp.ProfileReport(train)

pp_train
train = train.drop('user_answer',axis=1)

train = train.query('answered_correctly != -1').reset_index(drop=True)

train.info()
def missing(df):

    total = df.isnull().sum().sort_values(ascending = False)

    total = total[total>0]

    percent = df.isnull().sum().sort_values(ascending = False)/len(df)*100

    percent = percent[percent>0]

    return pd.concat([total, percent], axis=1, keys=['Total','Percentage'])

missing(train)
train['prior_question_had_explanation'] = train['prior_question_had_explanation'].fillna(train['prior_question_had_explanation'].mode()[0])

train['prior_question_elapsed_time'] = train['prior_question_elapsed_time'].fillna(train['prior_question_elapsed_time'].mean())

missing(train)
train.describe()
Counter(train['answered_correctly'])
train['prior_question_had_explanation'] = train['prior_question_had_explanation'].astype(float)

y = train['answered_correctly']

X = train.drop('answered_correctly', axis=1)
cat_col = ['user_id', 'content_type_id', 'task_container_id', 'prior_question_had_explanation']

lgb = LGBMClassifier(n_estimators = 200,class_weight="balanced", max_depth=-1,metric = 'auc',tree_learner = 'serial',learning_rate=0.08,

                     max_bin = 200,min_data_in_leaf = 80,num_leaves = 50, feature_fraction = 0.05,bagging_fraction = 0.4,random_state=42)  

fold = StratifiedKFold(n_splits = 5, shuffle =True)

pred = [];score_test =[];score_train =[];

for train_index , test_index in fold.split(X,y):

    X_train,X_test = X.iloc[train_index], X.iloc[test_index]

    y_train,y_test = y.iloc[train_index], y.iloc[test_index]

    lgb= lgb.fit(X_train, y_train,eval_set=(X_test , y_test),eval_metric='auc',verbose=50,categorical_feature=cat_col,early_stopping_rounds= 50)

    y_pred_train = lgb.predict_proba(X_train)[:,1]

    y_pred_test = lgb.predict_proba(X_test)[:,1]

    score_train.append(roc_auc_score( y_train,y_pred_train))

    score_test.append(roc_auc_score( y_test,y_pred_test))

print('\n')

print('Mean training AUC:',np.mean(score_train))

print('Mean testing AUC:',np.mean(score_test))
cat_col = ['user_id', 'content_type_id', 'task_container_id']

cat = CatBoostClassifier(bagging_temperature=0.717,border_count=198, depth=10, iterations=1500,

l2_leaf_reg=25,learning_rate=0.154, random_strength=0.003,eval_metric='AUC')

fold = StratifiedKFold(n_splits = 5, shuffle =True)

pred = [];score_test =[];score_train =[];

for train_index , test_index in fold.split(X,y):

    X_train,X_test = X.iloc[train_index], X.iloc[test_index]

    y_train,y_test = y.iloc[train_index], y.iloc[test_index]

    cat= cat.fit(X_train, y_train,eval_set=(X_test , y_test),verbose=50,cat_features=cat_col,early_stopping_rounds= 50)

    y_pred_train = cat.predict_proba(X_train)[:,1]

    y_pred_test = cat.predict_proba(X_test)[:,1]

    score_train.append(roc_auc_score( y_train,y_pred_train))

    score_test.append(roc_auc_score( y_test,y_pred_test))

print('\n')

print('Mean training AUC:',np.mean(score_train))

print('Mean testing AUC:',np.mean(score_test))
lgb = lgb.fit(X, y,eval_set=(X, y),eval_metric='auc',verbose=50,categorical_feature=cat_col,early_stopping_rounds= 50)

cat = cat.fit(X, y,verbose=50,cat_features=cat_col,early_stopping_rounds= 50)

cat_pred = cat.predict(X)

lgb_pred = lgb.predict(X)
blend = (0.6*lgb_pred) + (0.4*cat_pred)

roc_auc_score(y,blend)
iter_test = env.iter_test()
for (test, sample_prediction) in iter_test:

    X_test = test.drop([ 'prior_group_answers_correct','prior_group_responses'],axis = 1)

    X_test['prior_question_had_explanation'] = X_test['prior_question_had_explanation'].fillna(X_test['prior_question_had_explanation'].mode()[0])

    X_test['prior_question_elapsed_time'] = X_test['prior_question_elapsed_time'].fillna(X_test['prior_question_elapsed_time'].mean())

    X_test['prior_question_had_explanation'] = X_test['prior_question_had_explanation'].astype(float)

    pred_lgb = lgb.predict(X_test)

    pred_cat = cat.predict(X_test)

    X_test['answered_correctly'] = (0.6*pred_lgb) + (0.4*pred_cat)

    env.predict(X_test.loc[X_test['content_type_id'] == 0, ['row_id', 'answered_correctly']])