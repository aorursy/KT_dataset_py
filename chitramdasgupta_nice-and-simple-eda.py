import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style('dark')
import sklearn
import tensorflow as tf
from tensorflow import keras
read_num = 10 ** 6

train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', low_memory=False, nrows=read_num, 
                       dtype={'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',
                              'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 
                             'prior_question_had_explanation': 'boolean',})
train.shape
train.columns
train.describe()
train.isnull().sum()
train.groupby('answered_correctly').mean()['timestamp'].plot(kind='bar')
plt.ylabel('Mean timestamp')
plt.tight_layout()
sns.distplot(np.log1p(train['timestamp']), bins=40, kde=True)
plt.title('Distribution of timestamp values')
plt.tight_layout()
train['user_id'].unique().sum()
user_id_grp = train.groupby('user_id')
temp = user_id_grp['answered_correctly'].agg('count')
temp = dict(temp)
print('Minimum number of interactions from a single user', min(temp.values()))
print('Maximum number of interactions from a single user', max(temp.values()))
train['content_type_id'].value_counts()
train.groupby('content_type_id').count()['row_id'].plot(kind='bar')
plt.yscale('log')
plt.tight_layout()
train['task_container_id'].unique().sum()
train.groupby('answered_correctly').count()['row_id'].plot(kind='bar')
plt.ylabel('Number of answers')
plt.tight_layout()