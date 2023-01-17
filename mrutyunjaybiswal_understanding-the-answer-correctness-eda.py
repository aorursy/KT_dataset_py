# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
train = pd.read_csv('../input/riiid-test-answer-prediction/train.csv',

                   low_memory=False,

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

                         },

                   nrows=10**5)  # choose nrows value in 1e6 range at max unless you want your kernel to blow up

train
# knowing more about our train data

train.describe()
train.info()
print("Percentage Null values present in respective columns: ")

print(train.isnull().sum() * 100/len(train))
print(f"{train.user_id.nunique()} unique users are there")
train.user_id.value_counts()[:5]  # top 5 users with maximum content interaction
sample_user_id = 1283420



sample_chunk = train[train['user_id']==sample_user_id]

sample_chunk
# plot the continuous plot of timestamps

plt.plot(np.arange(0, len(sample_chunk)), sample_chunk['timestamp'])

plt.show()