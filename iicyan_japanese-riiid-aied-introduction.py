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
!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl > /dev/null
import sklearn.datasets as datasets

import pandas as pd



import matplotlib.pyplot as plt



import seaborn as sns

import datatable as dt

pd.options.display.float_format = '{:.20g}'.format


data_types = {

    'row_id': 'int32',

    'timestamp': 'int64',

    'user_id': 'int64',

    'content_id': 'int16',

    'content_type_id': 'int8',

    'task_container_id': 'int16',

    'user_answer': 'int8',

    'answered_correctly': 'int8',

    'prior_question_elapsed_time': 'float32',

    'prior_question_had_explanation': 'boolean'

}



train_df = dt.fread('../input/riiid-test-answer-prediction/train.csv').to_pandas()



# train_df = train_df.sample(len(train_df)//5,random_state=42)

for column, d_type in data_types.items():

    train_df[column] = train_df[column].astype(d_type) 
train_df.describe()
train_df.head()
questions_df = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')

lectures_df = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')

test_df = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')
questions_df.describe()
questions_df.head()
lectures_df.describe()
lectures_df.head()
test_df.describe()
test_df.head()