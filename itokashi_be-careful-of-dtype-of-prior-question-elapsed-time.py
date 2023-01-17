import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import seaborn as sns

import matplotlib.pyplot as plt
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
%%time

df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', dtype=data_types_dict)
df[df['user_id'] == 5382]
data_types_dict = {

    'row_id': 'int64',

    'timestamp': 'int64',

    'user_id': 'int32',

    'content_id': 'int16',

    'content_type_id': 'int8',

    'task_container_id': 'int16',

    'user_answer': 'int8',

    'answered_correctly': 'int8',

#     'prior_question_elapsed_time': 'float16',

    'prior_question_had_explanation': 'boolean'

}
%%time

df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', dtype=data_types_dict)
df[df['user_id'] == 5382]