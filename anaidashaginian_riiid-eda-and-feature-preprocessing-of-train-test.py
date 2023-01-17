# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%%time

dtypes = {
    "row_id": "int64",
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "boolean",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32", 
    "prior_question_had_explanation": "boolean"
}

data = pd.read_csv("../input/riiid-test-answer-prediction/train.csv", dtype=dtypes, nrows=50000000)

print("Train size:", data.shape)
data.head()
qdata = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
ldata = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')
test_data = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')
qdata.head()
ldata.head()
test_data.head()
data.row_id.value_counts()
data.timestamp.hist()
sns.distplot(data[data.user_id==data.user_id.unique()[0]].timestamp)
sns.distplot(data[data.user_id==data.user_id.unique()[1]].timestamp)
sns.distplot(data[data.user_id==data.user_id.unique()[2]].timestamp)
data[data.user_id==data.groupby('user_id').answered_correctly.sum().sort_values(ascending=False).index[0]].timestamp.hist()
data[data.user_id==data.groupby('user_id').answered_correctly.sum().sort_values(ascending=False).index[1]].timestamp.hist()
data.user_id.hist()
data.content_id.hist()
sns.countplot(data.content_type_id)
sns.countplot(data.user_answer)
sns.countplot(qdata.correct_answer)
sns.countplot(data.answered_correctly)
pd.Series(data.groupby('user_id').answered_correctly.sum().sort_values(ascending=False).iloc[:30], index=data.groupby('user_id').answered_correctly.sum().sort_values(ascending=False).iloc[:30].index).sort_values().plot(kind='barh')
data.groupby('user_id').answered_correctly.sum().median()
data.groupby('user_id').answered_correctly.sum().mean()
sums = data.groupby('user_id').answered_correctly.sum()
smart_users = sums[sums > sums.quantile(0.75)].index
data['is_smart'] = 0
data.loc[data.user_id.isin(smart_users), 'is_smart'] = 1
del sums
del smart_users
data.is_smart.value_counts()
sum_time = data.groupby('user_id').timestamp.sum()
all_sum = sum_time.sum()
data['sum_timestamp'] = data['user_id'].apply(lambda x: sum_time.loc[x]/all_sum)
del sum_time
del all_sum
mean0 = data[data.is_smart==0].sum_timestamp.mean()
mean1 = data[data.is_smart==1].sum_timestamp.mean()
mean0 / (mean0 + mean1), mean1 / (mean0 + mean1)
data.prior_question_elapsed_time.hist()
sns.countplot(data.prior_question_had_explanation)
traintest_users = list(set(test_data.user_id) & set(data.user_id))
traintest_users
len(traintest_users)
test_data['is_smart'] = 0
sum_time = test_data.groupby('user_id').timestamp.sum()
all_sum = sum_time.sum()
test_data['sum_timestamp'] = test_data['user_id'].apply(lambda x: sum_time.loc[x]/all_sum)
smart_guys = data[data.user_id.isin(traintest_users)].groupby('user_id')['is_smart'].max()
test_data.loc[test_data.user_id.isin(traintest_users), 'is_smart'] = test_data[test_data.user_id.isin(traintest_users)]['user_id'].apply(lambda x: smart_guys.loc[x])
test_data
data[['is_smart','sum_timestamp']].corr()