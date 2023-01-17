# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import timeit
train =  pd.read_csv('../input/train.csv', header=0)

test =  pd.read_csv('../input/test.csv', header=0)

test.columns = ['id','question1','question2']

test['qid1'] = 0

test['qid2'] = 0

test['is_duplicate'] = 999

df = pd.concat([train,test],0)
questions = pd.Series(list(set(df['question1']) | set(df['question2'])))

questions.fillna('',inplace=True)



q_all_count = pd.concat([df.groupby('question1').size(), df.groupby('question2').size()], 1).sum(1).astype(int)-1

q_all_count_df = q_all_count.to_frame().reset_index().rename(columns={'index':'question',0:'count'})

q_all_count_df.head()
train = train.merge(q_all_count_df.add_prefix('q1_'), left_on='question1', right_on='q1_question',how='left').drop('q1_question',1)

train = train.merge(q_all_count_df.add_prefix('q2_'), left_on='question2', right_on='q2_question',how='left').drop('q2_question',1)

train['q1_count'] = train['q1_count'].fillna(0).astype(int)

train['q2_count'] = train['q2_count'].fillna(0).astype(int)

train['q1_q2_count_diff'] = np.abs(train['q1_count']-train['q2_count'])
train.head(1)
test = test.merge(q_all_count_df.add_prefix('q1_'), left_on='question1', right_on='q1_question',how='left').drop('q1_question',1)

test = test.merge(q_all_count_df.add_prefix('q2_'), left_on='question2', right_on='q2_question',how='left').drop('q2_question',1)

test['q1_count'] = test['q1_count'].fillna(0).astype(int)

test['q2_count'] = test['q2_count'].fillna(0).astype(int)

test['q1_q2_count_diff'] = np.abs(test['q1_count']-train['q2_count'])
test.head(1)
corr_mat = train.corr()

corr_mat.head()

#more frequenct questions are more likely to be duplicates