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
train = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                   usecols=[1, 2, 3, 4, 7, 8, 9],

                   dtype={'timestamp': 'int64',

                          'user_id': 'int32',

                          'content_id': 'int16',

                          'content_type_id': 'int8',

                          'answered_correctly':'int8',

                          'prior_question_elapsed_time': 'float32',

                          'prior_question_had_explanation': 'boolean'}

                   )
train = train.sort_values(['timestamp'], ascending=True).reset_index(drop = True)
train.to_csv('train_time_ordered.csv', index=None)
print('done')