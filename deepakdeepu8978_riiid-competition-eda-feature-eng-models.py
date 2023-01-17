import time

import matplotlib.pyplot as plt 

import numpy as np

import pandas as pd

import seaborn as sns

color = sns.color_palette()

import os

       

import plotly.express as px 

import plotly.graph_objects as go

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit





import warnings

warnings.filterwarnings('ignore')
path = '/kaggle/input'



train = pd.read_csv(f'{path}/riiid-test-answer-prediction/train.csv', low_memory=False, nrows=5 * (10**5), 

                       dtype={'row_id': 'int64',

                              'timestamp': 'int64',

                              'user_id': 'int32',

                              'content_id': 'int16',

                              'content_type_id': 'int8',

                              'task_container_id': 'int16',

                              'user_answer': 'int8',

                              'answered_correctly': 'int8',

                              'prior_question_elapsed_time': 'float32', 

                              'prior_question_had_explanation': 'boolean',

                             }

                      )



test = pd.read_csv(f'{path}/riiid-test-answer-prediction/example_test.csv')

submit = pd.read_csv(f'{path}/riiid-test-answer-prediction/example_sample_submission.csv')

questions = pd.read_csv(f'{path}/riiid-test-answer-prediction/questions.csv')

lectures = pd.read_csv(f'{path}/riiid-test-answer-prediction/lectures.csv')

print('Train shapes: ', train.shape)

print('Test shapes: ', test.shape)
train.head()
fig,ax = plt.subplots(figsize=(12,8))

plt.hist(train['timestamp'], bins=40);

plt.xlabel('timestamp',fontsize=20)

plt.ylabel('count',fontsize=20)

ax.tick_params(labelsize=20)

plt.title('count of timestamp',fontsize=25)

plt.grid()

plt.ioff()
ds = train['user_id'].value_counts().reset_index()

ds.columns = ['user_id', 'count']

ds['user_id'] = ds['user_id'].astype(str) + '-'

ds = ds.sort_values(['count'])

top_40 = ds.tail(40)





fig,ax = plt.subplots(figsize=(12,8))

sns.barplot(top_40['count'],top_40['user_id'])
train.content_type_id.value_counts()
f,ax=plt.subplots(1,2,figsize=(18,8))

train['content_type_id'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Percentage content_type_id Distribution')

ax[0].set_ylabel('Count')

sns.countplot('content_type_id',data=train,ax=ax[1],order=train['content_type_id'].value_counts().index)

ax[1].set_title('Count of content_type_id')

plt.show()
train.task_container_id.value_counts()
train.user_answer.value_counts()
f,ax=plt.subplots(1,2,figsize=(18,8))

train['user_answer'].value_counts().plot.pie(explode=[0,0.1,0.1,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Percentage user_answer Distribution')

ax[0].set_ylabel('Count')

sns.countplot('user_answer',data=train,ax=ax[1],order=train['user_answer'].value_counts().index)

ax[1].set_title('Count of user_answer')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

train['answered_correctly'].value_counts().plot.pie(explode=[0,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Percentage Severity Distribution')

ax[0].set_ylabel('Count')

sns.countplot('answered_correctly',data=train,ax=ax[1],order=train['answered_correctly'].value_counts().index)

ax[1].set_title('Count of answered_correctly')

plt.show()
train['prior_question_had_explanation'].value_counts()
train = train.loc[train['answered_correctly'] != -1].reset_index(drop=True)

train['prior_question_had_explanation'] = train['prior_question_had_explanation'].fillna(value=False).astype(bool)



user_answers_ = train.groupby('user_id').agg({ 'answered_correctly': ['mean', 'count']}).copy()

user_answers_.columns = ['mean_user_accuracy', 'questions_answered']



content_answers_ = train.groupby('content_id').agg({'answered_correctly': ['mean', 'count']}).copy()

content_answers_.columns = ['mean_acc', 'questions_asked']



train = train.merge(user_answers_, how='left', on = 'user_id')

train = train.merge(content_answers_, how='left', on = 'content_id')
train = pd.merge(train,questions[['question_id','bundle_id','part']], left_on='user_id', right_on='question_id')

train.head()
grouped_df = train.groupby(["questions_answered"])["mean_user_accuracy"].aggregate("count").reset_index()



plt.figure(figsize=(12,8))

sns.pointplot(grouped_df['questions_answered'].values, grouped_df['mean_user_accuracy'].values, alpha=0.8, color=color[2])

plt.ylabel('questions_answered', fontsize=12)

plt.xlabel('mean_user_accuracy', fontsize=12)

plt.title("mean_user_accuracy wise questions_answered", fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
sns.jointplot(x=train.mean_acc.values,y=train.questions_asked.values,height=10)

plt.ylabel('mean_acc', fontsize=12)

plt.xlabel('questions_asked', fontsize=12)

plt.show()
columns = ['timestamp', 'user_id', 'content_id', 'content_type_id','answered_correctly',

       'task_container_id', 'prior_question_elapsed_time',

       'prior_question_had_explanation', 'part', 'mean_user_accuracy', 'questions_answered','mean_acc', 'questions_asked']
df = train[columns].copy()

df.info()
labels = []

values = []

for col in columns:

    labels.append(col)

    values.append(np.corrcoef(df[col].values, df.answered_correctly.values)[0,1])

corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})

corr_df = corr_df.sort_values(by='corr_values')



ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,40))

rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')

ax.set_yticks(ind)

ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')

ax.set_xlabel("Correlation coefficient")

ax.set_title("Correlation coefficient of the variables")

plt.show()
columns_f = ['timestamp', 'user_id', 'content_id', 'content_type_id',

       'task_container_id', 'prior_question_elapsed_time',

       'prior_question_had_explanation', 'part', 'mean_user_accuracy', 'questions_answered','mean_acc', 'questions_asked']
train.fillna(value = -1, inplace = True)
train.isnull().sum()
train_y = train['answered_correctly'].values

num_df = train[columns_f]

feat_name = num_df.columns.values



from sklearn import ensemble 

model = ensemble.RandomForestClassifier(n_estimators=25,max_depth=30, n_jobs=-1, random_state=0) 

model.fit(num_df,train_y)



importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)

indi = np.argsort(importances)[::-1][:20]



plt.figure(figsize=(12,12))

plt.title("Feature importances")

plt.bar(range(len(indi)), importances[indi], color=color[4], yerr=std[indi], align="center")

plt.xticks(range(len(indi)), feat_name[indi], rotation='vertical')

plt.xlim([-1, len(indi)])

plt.show()
import xgboost as xgb 



xgb_prames = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'silent': 1,

    'seed' : 0

}



dtrain = xgb.DMatrix(num_df, train_y, feature_names=num_df.columns.values)



model = xgb.train(dict(xgb_prames, silent=0), dtrain, num_boost_round=50)





fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
y = train['answered_correctly']

X = train.drop(['answered_correctly', 'user_answer'], axis=1)
def fast_auc(y_true, y_prob):

    """

    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013

    """

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    nfalse = 0

    auc = 0

    n = len(y_true)

    for i in range(n):

        y_i = y_true[i]

        nfalse += (1 - y_i)

        auc += y_i * nfalse

    auc /= (nfalse * (n - nfalse))

    return auc





def eval_auc(y_true, y_pred):

    """

    Fast auc eval function for lgb.

    """

    return 'auc', fast_auc(y_true, y_pred), True
import lightgbm as lgb



scores = []

feature_importance = pd.DataFrame()

models = []



params = {'num_leaves': 32,

          'max_bin': 300,

          'objective': 'binary',

          'max_depth': 13,

          'learning_rate': 0.03,

          "boosting_type": "gbdt",

          "metric": 'auc',

         }



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):

    print(f'Fold {fold_n} started at {time.ctime()}')

    X_train, X_valid = X[columns_f].iloc[train_index], X[columns_f].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    model = lgb.LGBMClassifier(**params, n_estimators=700, n_jobs = 1)

    model.fit(X_train, y_train, 

            eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=eval_auc,

            verbose=1000, early_stopping_rounds=10)

    score = max(model.evals_result_['valid_1']['auc'])

    

    models.append(model)

    scores.append(score)



    fold_importance = pd.DataFrame()

    fold_importance["feature"] = columns_f

    fold_importance["importance"] = model.feature_importances_

    fold_importance["fold"] = fold_n + 1

    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
feature_importance["importance"] /= 5

cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

    by="importance", ascending=False)[:50].index



best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



plt.figure(figsize=(16, 12));

sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

plt.title('LGB Features (avg over folds)');
import riiideducation

env = riiideducation.make_env()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    y_preds = []

    test_df = test_df.merge(user_answers_, how = 'left', on = 'user_id')

    test_df = test_df.merge(content_answers_, how = 'left', on = 'content_id')

    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value = False).astype(bool)

    test_df = test_df.loc[test_df['content_type_id'] == 0].reset_index(drop=True)

    test_df = pd.merge(test_df, questions[['question_id', 'bundle_id', 'part']], left_on='content_id', right_on='question_id', how='left')

    test_df.fillna(value = -1, inplace = True)



    for model in models:

        y_pred = model.predict_proba(test_df[columns_f], num_iteration=model.best_iteration_)[:, 1]

        y_preds.append(y_pred)



    y_preds = sum(y_preds) / len(y_preds)

    test_df['answered_correctly'] = y_preds

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])