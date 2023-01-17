# This is how I built my validation set. I'm curious if you use the same? If not, say what set you use.
# IMPORTS

import numpy as np
import pandas as pd
import os
from datetime import datetime
import gc
# SETTINGS

# Path to train.csv and valid.csv
path_in = '../input/train.csv'
path_out = '../input/valid.csv'

# Names of columns to load from train set
train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']

# Types of columns to load from trains set
dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

# Limit dates for the training and test set
test_start_date = "2017-11-09 04:00:00"
test_end_date = "2017-11-09 16:00:00"

test_start_date = datetime.strptime(test_start_date, '%Y-%m-%d %H:%M:%S')
test_end_date = datetime.strptime(test_end_date, '%Y-%m-%d %H:%M:%S')
def buildValidationSet(df):
    print("Converting to datetime...")
    df['click_time'] = pd.to_datetime(df['click_time'])
    print("Filtration of dataset...")
    df = df[(df['click_time'] < test_end_date) & (df['click_time'] >= test_start_date)]
    print("Number of unique entries per each hour:")
    print(df.click_time.dt.hour.value_counts())
    return df
# LOAD DATA
df_train = pd.read_csv(path_in, usecols=train_columns, dtype=dtypes)
df_train = buildValidationSet(df_train)
gc.collect()
# this line is, of course, to be uncommented
# df_train.to_csv(path_out)
gc.collect()