# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style('dark')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_types_dict = {

    'row_id': 'int64',

    'timestamp': 'int64',

    'user_id': 'int32',

    'content_id': 'int16',

    'content_type_id': 'int8',

     'task_container_id': 'int16',

     'user_answer': 'int8',

    'answered_correctly': 'int8',

    'prior_question_elapsed_time': 'float16',

    'prior_question_had_explanation': 'boolean'

}
train = pd.read_csv('../input/riiid-test-answer-prediction/train.csv',

                       usecols = data_types_dict.keys(),

                       dtype=data_types_dict)

lectures = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')

questions = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')

example_test = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')
train.head()

    
lectures.head()
questions.head()
example_test.head()
train.dtypes
train.describe()
train.info()
train.isnull().sum()
cols = train.columns

for col in cols: 

    print('Unique values in {} :  {}'.format(col,train[col].nunique()))
train.columns
plt.figure(figsize=(12,8))

train['timestamp'].hist(bins=100)

plt.show()
plt.figure(figsize=(12,8))

train['content_type_id'].hist(bins=100)

plt.show()



cols = ['timestamp', 'user_id', 'content_id',

       'task_container_id']

sns.set_style('darkgrid')

fig,ax=plt.subplots(figsize=(18,12))

for i in range(len(cols)):



    plt.subplot(2, 2, i+1)

    sns.distplot(train[cols[i]],color='blue')
cols = ['user_answer', 'answered_correctly']

sns.set_style('darkgrid')

fig,ax=plt.subplots(figsize=(12,8))

for i in range(len(cols)):



    plt.subplot(1, 2, i+1)

    sns.countplot(train[cols[i]])


plt.figure(figsize=(12,8))

sns.countplot(train['user_answer'], hue=train['answered_correctly'])

plt.show()