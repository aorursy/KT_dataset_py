# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import sys

import matplotlib.pyplot as plt

from catboost import CatBoostRegressor

from tqdm import tqdm

from sklearn.model_selection import train_test_split



# Any results you write to the current directory are saved as output.
# constants

trace_length = 150000

step_size = 25000

expected_passes = 629145480//step_size
columns = ['freq_'+str(i) for i in range(1200)]

transformed_data = []

target = []

previous_df = pd.DataFrame()

chunk_no = 0

window_no = 0

for train_df in tqdm(pd.read_csv('../input/train.csv', chunksize =2* 10 ** 5, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})):

    chunk_no += 1

#     print ("processing chunk number ", chunk_no)

    if not previous_df.empty:

        train_df = pd.concat([previous_df, train_df], axis=0)

    start_index = min(train_df.index.values)

    while start_index + trace_length <= max(train_df.index.values):

        data_list = []

        interim_index = start_index

        for _ in range(3):

            power_spectrum = np.fft.fft(train_df.loc[interim_index:interim_index+50000, 'acoustic_data'].values)

            power_spectrum = np.absolute(power_spectrum)

            data_list.extend(list(power_spectrum[:10000][::25]))

            interim_index += 50000

        transformed_data.append(data_list)

        target.append(np.mean(train_df.loc[start_index:start_index+150000, 'time_to_failure'].values))

        start_index += step_size

        window_no += 1

#         sys.stdout.flush()

#         print ("windows processed ", window_no)

#     if window_no > 10000:

#         print ("10000 windows processed")

#         break

        

        

    previous_df = train_df.loc[start_index:, :]

X_train = pd.DataFrame(columns=columns, data=transformed_data)

y_train = pd.Series(data=target)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, shuffle=True, test_size=0.2, 

                                                    random_state=42)
params = {

    'iterations': 2000,

    'learning_rate': 0.1,

    'eval_metric': 'MAE',

    'random_seed': 42,

    'logging_level': 'Silent',

    'use_best_model': False

}

earlystop_params = params.copy()

earlystop_params.update({

    'od_type': 'Iter',

    'od_wait': 100

})

earlystop_model = CatBoostRegressor(**earlystop_params)
earlystop_model.fit(

    X_train, y_train,

    eval_set=(X_test, y_test),

    logging_level='Verbose',  # you can uncomment this for text output

    plot=True

)
transformed_data = []

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

for i, seg_id in enumerate(tqdm(submission.index)):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    interim_index = 0

    data_list = []

    for _ in range(3):

        power_spectrum = np.fft.fft(seg.loc[interim_index:interim_index+50000, 'acoustic_data'].values)

        power_spectrum = np.absolute(power_spectrum)

        data_list.extend(list(power_spectrum[:10000][::25]))

        interim_index += 50000

    transformed_data.append(data_list)

test_transformed = pd.DataFrame(columns=columns, data=transformed_data, index=submission.index)
predictions = earlystop_model.predict(test_transformed)

submission['time_to_failure'] = predictions
submission.to_csv('submission.csv')