import riiideducation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
env = riiideducation.make_env()
train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',
                   usecols=[1, 2, 3, 4, 5, 7, 8, 9],
                   dtype={'timestamp': 'int64',
                          'user_id': 'int32',
                          'content_id': 'int16',
                          'content_type_id': 'int8',
                          'task_container_id': 'int16',
                          'answered_correctly':'int8',
                          'prior_question_elapsed_time': 'float32',
                          'prior_question_had_explanation': 'boolean'}
                   )
train.info()
train
#reading in question df
questions_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv',                         
                            usecols=[0, 3],
                            dtype={'question_id': 'int16',
                              'part': 'int8'}
                          )
questions_df
train = train[train.content_type_id == False].sort_values('timestamp').reset_index(drop = True)
elapsed_mean = train.prior_question_elapsed_time.mean()
elapsed_mean
group1 = train.loc[(train.content_type_id == False), ['task_container_id', 'user_id']].groupby(['task_container_id']).agg(['count'])
group1.columns = ['avg_questions']
group2 = train.loc[(train.content_type_id == False), ['task_container_id', 'user_id']].groupby(['task_container_id']).agg(['nunique'])
group2.columns = ['avg_questions']
group3 = group1 / group2
group3['avg_questions_seen'] = group3.avg_questions.cumsum()
print('The amount of questions seen by the average user:')
group3.iloc[0].avg_questions_seen
results_u_final = train.loc[train.content_type_id == False, ['user_id','answered_correctly']].groupby(['user_id']).agg(['mean', 'count'])
results_u_final.columns = ['answered_correctly_user','answered_user']
results_u_final.answered_correctly_user.describe()

results_u2_final = train.loc[train.content_type_id == False, ['user_id','prior_question_had_explanation']].groupby(['user_id']).agg(['mean'])
results_u2_final.columns = ['explanation_mean_user']
results_u2_final.explanation_mean_user.describe()
train = pd.merge(train, questions_df, left_on = 'content_id', right_on = 'question_id', how = 'left')
results_q_final = train.loc[train.content_type_id == False, ['question_id','answered_correctly']].groupby(['question_id']).agg(['mean'])
results_q_final.columns = ['quest_pct']
results_q_final.quest_pct.describe()
results_q2_final = train.loc[train.content_type_id == False, ['question_id','part']].groupby(['question_id']).agg(['count'])
results_q2_final.columns = ['count']
question2 = pd.merge(questions_df, results_q_final, left_on = 'question_id', right_on = 'question_id', how = 'left')
question2 = pd.merge(question2, results_q2_final, left_on = 'question_id', right_on = 'question_id', how = 'left')
question2.quest_pct = round(question2.quest_pct,5)
question2
figure=plt.subplots(figsize=(20,20))
plt.scatter(x = results_u_final.answered_user, y=results_u_final.answered_correctly_user)
plt.axhline(train['answered_correctly'].mean(), color='k', linestyle='dashed', linewidth=3)

plt.title("Fraction of the user's answers that are correct vs. Number of questions answered by the user", weight='bold')
plt.text(15000, 0.64, 'Fraction of answers that are correct: {:.2f}'.format(train['answered_correctly'].mean()))
plt.show()
train.loc[(train.timestamp == 0)].answered_correctly.mean()
train.loc[(train.timestamp != 0)].answered_correctly.mean()
prior_mean_user = results_u2_final.explanation_mean_user.mean()
prior_mean_user
train.drop(['timestamp', 'content_type_id', 'question_id', 'part'], axis=1, inplace=True)
print('The old length of the training set:')
print(len(train))
validation = train.groupby('user_id').tail(10)
train = train[~train.index.isin(validation.index)]
print('The length of the training set plus the length of the validation set:')
print(len(train) + len(validation))
results_u_val = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean'])
results_u_val.columns = ['answered_correctly_user']

results_u2_val = train[['user_id','prior_question_had_explanation']].groupby(['user_id']).agg(['mean'])
results_u2_val.columns = ['explanation_mean_user']
X = train.groupby('user_id').tail(30)
train = train[~train.index.isin(X.index)]
print('The length of the training set plus the length of the validation set plus the length of the set to be discarded:')
print(len(X) + len(validation)+ len(train))
results_u_X = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean'])
results_u_X.columns = ['answered_correctly_user']

results_u2_X = train[['user_id','prior_question_had_explanation']].groupby(['user_id']).agg(['mean'])
results_u2_X.columns = ['explanation_mean_user']
del(train)
X = pd.merge(X, group3, left_on=['task_container_id'], right_index= True, how="left")
X = pd.merge(X, results_u_X, on=['user_id'], how="left")
X = pd.merge(X, results_u2_X, on=['user_id'], how="left")
validation = pd.merge(validation, group3, left_on=['task_container_id'], right_index= True, how="left")
validation = pd.merge(validation, results_u_val, on=['user_id'], how="left")
validation = pd.merge(validation, results_u2_val, on=['user_id'], how="left")
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

X.prior_question_had_explanation.fillna(False, inplace = True)
validation.prior_question_had_explanation.fillna(False, inplace = True)

validation["prior_question_had_explanation_enc"] = lb_make.fit_transform(validation["prior_question_had_explanation"])
X["prior_question_had_explanation_enc"] = lb_make.fit_transform(X["prior_question_had_explanation"])
content_mean = question2.quest_pct.mean()

question2.quest_pct.mean()
#filling questions with no info with a new value
question2.quest_pct = question2.quest_pct.mask((question2['count'] < 3), .65)


#filling very hard new questions with a more reasonable value
question2.quest_pct = question2.quest_pct.mask((question2.quest_pct < .2) & (question2['count'] < 21), .2)

#filling very easy new questions with a more reasonable value
question2.quest_pct = question2.quest_pct.mask((question2.quest_pct > .95) & (question2['count'] < 21), .95)
X = pd.merge(X, question2, left_on = 'content_id', right_on = 'question_id', how = 'left')
validation = pd.merge(validation, question2, left_on = 'content_id', right_on = 'question_id', how = 'left')
y = X['answered_correctly']
X = X.drop(['answered_correctly'], axis=1)
X.head()

y_val = validation['answered_correctly']
X_val = validation.drop(['answered_correctly'], axis=1)
X = X[['answered_correctly_user', 'explanation_mean_user', 'quest_pct', 'avg_questions_seen',
       'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part']]
X_val = X_val[['answered_correctly_user', 'explanation_mean_user', 'quest_pct', 'avg_questions_seen',
       'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part']]
X['answered_correctly_user'].fillna(0.65,  inplace=True)
X['explanation_mean_user'].fillna(prior_mean_user,  inplace=True)
X['quest_pct'].fillna(content_mean, inplace=True)

X['part'].fillna(4, inplace = True)
X['avg_questions_seen'].fillna(1, inplace = True)
X['prior_question_elapsed_time'].fillna(elapsed_mean, inplace = True)
X['prior_question_had_explanation_enc'].fillna(0, inplace = True)
X_val['answered_correctly_user'].fillna(0.65,  inplace=True)
X_val['explanation_mean_user'].fillna(prior_mean_user,  inplace=True)
X_val['quest_pct'].fillna(content_mean,  inplace=True)

X_val['part'].fillna(4, inplace = True)
X['avg_questions_seen'].fillna(1, inplace = True)
X_val['prior_question_elapsed_time'].fillna(elapsed_mean, inplace = True)
X_val['prior_question_had_explanation_enc'].fillna(0, inplace = True)
import lightgbm as lgb

lgb_train = lgb.Dataset(X, y, categorical_feature = ['part', 'prior_question_had_explanation_enc'],free_raw_data=False)
lgb_eval = lgb.Dataset(X_val, y_val, categorical_feature = ['part', 'prior_question_had_explanation_enc'], reference=lgb_train, free_raw_data=False)
### import lightgbm as lgb
params = {
        'num_leaves': 161,
        'boosting_type': 'gbdt',
        'max_bin': 890,
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 12,
        'min_child_weight': 11,
        'feature_fraction': 0.6903098140467137,
        'bagging_fraction': 0.9267405716419829,
        'bagging_freq': 7,
        'min_child_samples': 77,
        'lambda_l1': 0.02267578846472961,
        'lambda_l2': 9.722845458292198e-08,
        'early_stopping_rounds': 10
        }
lgb_train = lgb.Dataset(X, y, categorical_feature = ['part', 'prior_question_had_explanation_enc'])
lgb_eval = lgb.Dataset(X_val, y_val, categorical_feature = ['part', 'prior_question_had_explanation_enc'], reference=lgb_train)
model = lgb.train(
    params, lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    verbose_eval=1000,
    num_boost_round=2000
)
import matplotlib.pyplot as plt
import seaborn as sns
lgb.plot_importance(model)
plt.show()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:
    test_df['task_container_id'] = test_df.task_container_id.mask(test_df.task_container_id > 9999, 9999)
    test_df = pd.merge(test_df, group3, left_on=['task_container_id'], right_index= True, how="left")
    test_df = pd.merge(test_df, question2, left_on = 'content_id', right_on = 'question_id', how = 'left')
    test_df = pd.merge(test_df, results_u_final, on=['user_id'],  how="left")
    test_df = pd.merge(test_df, results_u2_final, on=['user_id'],  how="left")
    test_df['answered_correctly_user'].fillna(0.65,  inplace=True)
    test_df['explanation_mean_user'].fillna(prior_mean_user,  inplace=True)
    test_df['quest_pct'].fillna(content_mean,  inplace=True)

    test_df['part'].fillna(4, inplace = True)
    test_df['avg_questions_seen'].fillna(1, inplace = True)
    test_df['prior_question_elapsed_time'].fillna(elapsed_mean, inplace = True)
    test_df['prior_question_had_explanation'].fillna(False, inplace=True)
    test_df["prior_question_had_explanation_enc"] = lb_make.fit_transform(test_df["prior_question_had_explanation"])
    
    test_df['answered_correctly'] =  model.predict(test_df[['answered_correctly_user', 'explanation_mean_user', 'quest_pct', 'avg_questions_seen',
                                                            'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part']])
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])