
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


train_df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', nrows=1000000)
lectures = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')
questions = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
example_test = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')
example_sample_submission = pd.read_csv('../input/riiid-test-answer-prediction/example_sample_submission.csv')
from datetime import datetime
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], unit='ms',origin='2017-1-1')
train_df['month']=(train_df.timestamp.dt.month)
train_df['day']=(train_df.timestamp.dt.day)
train_df=train_df.drop(columns=['timestamp'])
train_df['task_container_id'] = (
    train_df
    .groupby('user_id')['task_container_id']
    .transform(lambda x : pd.factorize(x)[0])
    .astype('int16')
)
train_df = train_df.sort_values(by=['user_id','row_id'])
train_df