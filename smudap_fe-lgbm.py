import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder
X = pd.read_csv('../input/kaggledays-warsaw/train.csv', sep="\t", index_col='id')
Y = pd.read_csv('../input/kaggledays-warsaw/test.csv', sep="\t", index_col='id')
sub = pd.read_csv('../input/kaggledays-warsaw/sample_submission.csv', index_col='id')
leaked = pd.read_csv('../input/kaggledays-warsaw/leaked_records.csv', index_col='id')
def process_question_id(data):
    data_copy = data.copy()
    question_id_counts = pd.DataFrame(data_copy['question_id'].value_counts()).reset_index().\
        rename(columns={'question_id': 'question_id_count', 'index': 'question_id'})
    data_copy = data_copy.merge(question_id_counts, how='left', on='question_id')
    return data_copy

def process_subreddit(data):
    data_copy = data.copy()
    subreddit_count = data_copy.groupby('subreddit')['question_score'].agg(lambda x: x.count()).\
    reset_index().rename(columns={'question_score': 'subreddit_count'})
    subreddit_unique_count = data_copy.groupby('subreddit')['question_score'].agg(lambda x: x.nunique()).\
        reset_index().rename(columns={'question_score': 'subreddit_unique_count'})
    subreddit_qs_mean = data_copy.groupby('subreddit')['question_score'].agg(lambda x: np.mean(x.unique())).\
        reset_index().rename(columns={'question_score': 'subreddit_qs_mean'})
    subreddit_qs_std = data_copy.groupby('subreddit')['question_score'].agg(lambda x: np.std(x.unique())).\
        reset_index().rename(columns={'question_score': 'subreddit_qs_std'})
    subreddit_qs_min = data_copy.groupby('subreddit')['question_score'].agg(lambda x: np.min(x.unique())).\
        reset_index().rename(columns={'question_score': 'subreddit_qs_min'})
    subreddit_qs_q1 = data_copy.groupby('subreddit')['question_score'].agg(lambda x: np.percentile(x.unique(), 25)).\
        reset_index().rename(columns={'question_score': 'subreddit_qs_q1'})
    subreddit_qs_q2 = data_copy.groupby('subreddit')['question_score'].agg(lambda x: np.percentile(x.unique(), 50)).\
        reset_index().rename(columns={'question_score': 'subreddit_qs_q2'})
    subreddit_qs_q3 = data_copy.groupby('subreddit')['question_score'].agg(lambda x: np.percentile(x.unique(), 75)).\
        reset_index().rename(columns={'question_score': 'subreddit_qs_q3'})
    subreddit_qs_max = data_copy.groupby('subreddit')['question_score'].agg(lambda x: np.max(x.unique())).\
        reset_index().rename(columns={'question_score': 'subreddit_qs_max'})
    subreddit_stats = subreddit_count.merge(subreddit_unique_count, how='left', on='subreddit')
    subreddit_stats = subreddit_stats.merge(subreddit_qs_mean, how='left', on='subreddit')
    subreddit_stats = subreddit_stats.merge(subreddit_qs_std, how='left', on='subreddit')
    subreddit_stats = subreddit_stats.merge(subreddit_qs_min, how='left', on='subreddit')
    subreddit_stats = subreddit_stats.merge(subreddit_qs_q1, how='left', on='subreddit')
    subreddit_stats = subreddit_stats.merge(subreddit_qs_q2, how='left', on='subreddit')
    subreddit_stats = subreddit_stats.merge(subreddit_qs_q3, how='left', on='subreddit')
    subreddit_stats = subreddit_stats.merge(subreddit_qs_max, how='left', on='subreddit')
    subreddit_stats
    data_copy = data_copy.merge(subreddit_stats, how='left', on='subreddit')
    data_copy['qs_better_subreddit_q1'] = (data_copy['question_score'] <= data_copy['subreddit_qs_q1']).astype('int')
    data_copy['qs_better_subreddit_q2'] = (data_copy['question_score'] <= data_copy['subreddit_qs_q2']).astype('int')
    data_copy['qs_better_subreddit_q3'] = (data_copy['question_score'] <= data_copy['subreddit_qs_q3']).astype('int')
    data_copy['qs_better_subreddit_mean'] = (data_copy['question_score'] <= data_copy['subreddit_qs_mean']).astype('int')
    return data_copy

def process_time(data):
    data_copy = data.copy()
    data_copy['response_time'] = data_copy['answer_utc'] - data_copy['question_utc']
    response_time_stats = data_copy.groupby('subreddit')['response_time'].describe().reset_index().drop('count', axis=1)
    response_time_stats.columns = ['subreddit', 'subreddit_rt_mean', 'subreddit_rt_std', 'subreddit_rt_min',
                                   'subreddit_rt_q1', 'subreddit_rt_q2', 'subreddit_rt_q3', 'subreddit_rt_max']
    data_copy = data_copy.merge(response_time_stats, how='left', on='subreddit')
    data_copy['rt_faster_subreddit_q1'] = (data_copy['response_time'] <= data_copy['subreddit_rt_q1']).astype('int')
    data_copy['rt_faster_subreddit_q2'] = (data_copy['response_time'] <= data_copy['subreddit_rt_q2']).astype('int')
    data_copy['rt_faster_subreddit_q3'] = (data_copy['response_time'] <= data_copy['subreddit_rt_q3']).astype('int')
    data_copy['rt_faster_subreddit_mean'] = (data_copy['response_time'] <= data_copy['subreddit_rt_mean']).astype('int')
    question_time = pd.to_datetime(data['question_utc'], unit='s')
    data_copy['question_day'] = question_time.dt.day
    data_copy['question_hour'] = question_time.dt.hour
    answer_time = pd.to_datetime(data['answer_utc'], unit='s')
    data_copy['answer_day'] = answer_time.dt.day
    data_copy['answer_hour'] = answer_time.dt.hour
    return data_copy

def process_target(data):
    data_copy = data.copy()
    target_mean = data_copy.groupby('subreddit')['answer_score'].mean().reset_index().\
        rename(columns={'answer_score': 'target_subreddit_mean'})
    sub_data = data_copy[['subreddit', 'question_score', 'answer_score']].copy()
    sub_data['subreddit_diff'] = sub_data['question_score'] - sub_data['answer_score']
    target_diff_mean = sub_data.groupby('subreddit')['subreddit_diff'].mean().reset_index().\
        rename(columns={'subreddit_diff': 'target_subreddit_diff_mean'})
    sub_data['subreddit_ratio'] = sub_data['question_score'] / sub_data['answer_score']
    target_ratio_mean = sub_data.groupby('subreddit')['subreddit_ratio'].mean().reset_index().\
        rename(columns={'subreddit_ratio': 'target_subreddit_ratio_mean'})
    target_stats = target_mean.merge(target_diff_mean, how='left', on='subreddit')
    target_stats = target_stats.merge(target_ratio_mean, how='left', on='subreddit')
    return target_stats

def merge_target(data, target_data):
    data_copy = data.copy()
    data_copy = data_copy.merge(target_data, how='left', on='subreddit')
    return data_copy

def process_question_answer_text(data):
    data_copy = data.copy()
    new_data = pd.DataFrame()
    new_data['question_signs'] = data['question_text'].apply(lambda x: len(x))
    new_data['answer_signs'] = data['answer_text'].apply(lambda x: len(x))
    new_data['question_words'] = data['question_text'].apply(lambda x: len(x.split()))
    new_data['answer_words'] = data['answer_text'].apply(lambda x: len(x.split()))
    new_data['diff_letters'] = new_data['question_signs'] - new_data['answer_signs']
    new_data['diff_words'] = new_data['question_words'] - new_data['answer_words']
    new_data['question_link'] = data['question_text'].apply(lambda x: ("www" in x) or ("http" in x)).astype('int')
    new_data['answer_link'] = data['answer_text'].apply(lambda x: ("www" in x) or ("http" in x)).astype('int')
    new_data['question_nb_big_letters'] = data['question_text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x))
    new_data['answer_nb_big_letters'] = data['answer_text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x))
    new_data['question_letters'] = data['question_text'].apply(lambda x: sum(1 for c in x if c.isalpha()) / len(x))
    new_data['answer_letters'] = data['answer_text'].apply(lambda x: sum(1 for c in x if c.isalpha()) / len(x))
    data_copy = pd.concat([data_copy, new_data], axis=1)
    return data_copy
X_new = X.copy()
X_new = process_question_id(X_new)
X_new = process_subreddit(X_new)
X_new = process_time(X_new)
X_new = merge_target(X_new, process_target(X))
X_new = process_question_answer_text(X_new)
Y_new = Y.copy()
Y_new = process_question_id(Y_new)
Y_new = process_subreddit(Y_new)
Y_new = process_time(Y_new)
Y_new = merge_target(Y_new, process_target(X))
Y_new = process_question_answer_text(Y_new)
X_new.shape, Y_new.shape
X_new.head()
Y_new.head()
train = X_new.drop(['question_id', 'question_text', 'answer_text'], axis=1)
test = Y_new.drop(['question_id', 'question_text', 'answer_text'], axis=1)
le = LabelEncoder()
le.fit(train['subreddit'])
train['subreddit'] = le.transform(train['subreddit'])
test['subreddit'] = le.transform(test['subreddit'])
train['answer_score'] = np.log1p(train['answer_score'])
np.random.seed(7)
questions = X_new['question_id'].unique()
train_questions = np.random.choice(questions, int(0.8*len(questions)), replace=False)
train_observations = X_new['question_id'].isin(train_questions)
train_data = lgbm.Dataset(train.drop(['answer_score'],axis=1).loc[train_observations,:], label=train.loc[train_observations, 'answer_score'])
val_data = lgbm.Dataset(train.drop(['answer_score'],axis=1).loc[~train_observations,:], label=train.loc[~train_observations, 'answer_score'])

params = {'objective': 'regression',
          'learning_rate': 0.015,
          'max_bin': 819,
          'num_leaves': 5046,
          'min_data_in_leaf': 175,
          'num_boost_round': 200,
          'metric': 'rmse'
    }    
# Train the model    
val_model = lgbm.train(params, train_data, valid_sets=[train_data, val_data])
full_train_data = lgbm.Dataset(train.drop(['answer_score'], axis=1), label=train.loc[:, 'answer_score'])

params = {'objective': 'regression',
          'learning_rate': 0.015,
          'max_bin': 819,
          'num_leaves': 5046,
          'min_data_in_leaf': 175,
          'num_boost_round': 200,
          'metric': 'rmse',
          'verbose': 1
    }    
# Train the model    
final_model = lgbm.train(params, full_train_data, valid_sets=[full_train_data])
preds = final_model.predict(test)
preds = np.expm1(preds)
magic_coef = 0.8
sub['answer_score'] = preds * magic_coef
sub.loc[leaked.index.tolist()] = leaked
sub.head()
sub.to_csv('submission.csv')
