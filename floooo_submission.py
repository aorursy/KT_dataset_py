import datetime



import numpy as np

import pandas as pd
from pathlib import Path



INPUT_PATH = Path('..', 'input', 'nyc-taxi-trip-duration')

TRAIN_PATH = Path(INPUT_PATH, 'train.zip')

TEST_PATH = Path(INPUT_PATH, 'test.zip')

SAMPLE_PATH = Path(INPUT_PATH, 'sample_submission.zip')
df_train = pd.read_csv(TRAIN_PATH, index_col='id',

                       parse_dates=['pickup_datetime', 'dropoff_datetime'])
submission = pd.read_csv(SAMPLE_PATH, index_col='id')

submission['trip_duration'] = df_train['trip_duration'].mean()

submission.head()
now = datetime.datetime.now().strftime('%Y%m%d_%H_%M')

filename = f'submission_{now}.csv'

submission.to_csv(filename)