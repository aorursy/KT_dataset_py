# https://www.kaggle.com/onodera/riiid-read-csv-in-cudf



import sys

!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/

import cudf
import pandas as pd

#import dask.dataframe as dd

#!pip install modin > /dev/null

#import modin.pandas as modin

!pip install feather-format > /dev/null

import feather

import joblib
from time import time

from contextlib import contextmanager



@contextmanager

def timer(name,times):

    t0 = time()

    yield

    t1 = time() - t0

    times.append(t1)

    print(f'[{name}] done in {t1:.2f} s')

    
pandas_times = []

feather_times = []

joblib_times = []

cudf_times = []



nrows = 10**7

with timer('pd.read_csv', pandas_times):

    df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                        #nrows=nrows,

                        usecols=[1, 2, 3, 4, 7, 8, 9],

                        dtype={'timestamp': 'int64',

                              'user_id': 'int32',

                              'content_id': 'int16',

                              'content_type_id': 'int8',

                              'answered_correctly':'int8',

                              'prior_question_elapsed_time': 'float32',

                              'prior_question_had_explanation': 'boolean'}

                       )

    

df.to_feather('./train.feather')

with timer('pd.read_feather', feather_times):

    df = pd.read_feather('./train.feather')

    

joblib.dump(df, './train.sav', compress=3)

with timer('joblib.load', joblib_times):

    df = joblib.load('./train.sav')

    

with timer('cudf.read_csv', cudf_times):

    df = cudf.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                        #nrows=nrows,

                        usecols=[1, 2, 3, 4, 7, 8, 9],

                        dtype={'timestamp': 'int64',

                              'user_id': 'int32',

                              'content_id': 'int16',

                              'content_type_id': 'int8',

                              'answered_correctly':'int8',

                              'prior_question_elapsed_time': 'float32',

                              'prior_question_had_explanation': 'boolean'}

                       )

import matplotlib.pyplot as plt



#plt.rcdefaults()

fig, ax = plt.subplots()



x = [1, 2, 3, 4]

y = [pandas_times[0], feather_times[0], joblib_times[0], cudf_times[0]]

label = ['pandas', 'feather', 'joblib', 'cuDF']



ax.barh(x, y, align='center')

ax.set_yticks(x)

ax.set_yticklabels(label)



ax.set_xlabel('time (s)')

ax.set_title('How long time does it take to read csv?')



plt.show()