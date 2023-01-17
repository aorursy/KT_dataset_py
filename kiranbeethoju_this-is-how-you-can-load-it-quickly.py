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
import dask.dataframe as dd
df = dd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',low_memory=False,
                dtype={'row_id': 'int64',
                        'timestamp': 'int64',
                        'user_id': 'int32',
                        'content_id':'int16',
                       'content_type_id': 'int8',
                       'task_container_id':'int16',
                       'user_answer': 'int8',
                       'answered_correctly':'int8',
                       'prior_question_elapsed_time':'float32',
                       'prior_question_had_explanation':'boolean'

                      }
                )
df.dtypes
train = df
!pip install chart_studio
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
train.columns
content_acc = train.query('answered_correctly != -1').groupby('content_id')['answered_correctly'].mean().to_dict()
import riiideducation

# You can only call make_env() once, so don't lose it!
env = riiideducation.make_env()
iter_test = env.iter_test()

def add_content_acc(x):
    if x in content_acc.keys():
        return content_acc[x]
    else:
        return 0.5


test = pd.read_csv("/kaggle/input/riiid-test-answer-prediction/example_test.csv")
for (test, sample_prediction_df) in iter_test:
    test['answered_correctly'] = test['content_id'].apply(add_content_acc).values
    env.predict(test.loc[test['content_type_id'] == 0, ['row_id', 'answered_correctly']])
test
