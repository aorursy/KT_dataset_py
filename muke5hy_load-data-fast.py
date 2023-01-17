import os

import os

import json

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data_path = '/kaggle/input/data-science-bowl-2019/'

for dirname, _, filenames in os.walk(data_path):

    for filename in filenames:

        print(os.path.join(dirname, filename))
keep_cols = ['event_id', 'game_session', 'installation_id', 'event_count', 'event_code', 'title', 'game_time', 'type', 'world','event_data']
%%time 



try:

    train = pd.read_pickle(f'train.pkl')

    test = pd.read_pickle(f'test.pkl')

    specs = pd.read_pickle(f'specs.pkl')

    train_labels = pd.read_pickle(f'train_labels.pkl')

    sample_submission = pd.read_pickle(f'sample_submission.pkl')

    print(f'Reading from pkl files')

except (OSError, IOError) as e:

    train = pd.read_csv(f'{data_path}train.csv')

    test = pd.read_csv(f'{data_path}test.csv')

    specs = pd.read_csv(f'{data_path}specs.csv')

    train_labels = pd.read_csv(f'{data_path}train_labels.csv')

    sample_submission = pd.read_csv(f'{data_path}sample_submission.csv')

    

    # uncomment below when using.   

#     train.to_pickle(f'train.pkl')

#     test.to_pickle(f'test.pkl')

#     specs.to_pickle(f'specs.pkl')

#     train_labels.to_pickle(f'train_labels.pkl')

#     sample_submission.to_pickle(f'sample_submission.pkl')

    print(f'Reading from CSV files')
%%time 



train.to_pickle(f'train.pkl')

test.to_pickle(f'test.pkl')

specs.to_pickle(f'specs.pkl')

train_labels.to_pickle(f'train_labels.pkl')

sample_submission.to_pickle(f'sample_submission.pkl')

print(f'Reading from CSV files')
%%time 



train = pd.read_pickle(f'train.pkl')

test = pd.read_pickle(f'test.pkl')

specs = pd.read_pickle(f'specs.pkl')

train_labels = pd.read_pickle(f'train_labels.pkl')

sample_submission = pd.read_pickle(f'sample_submission.pkl')

%%time 



try:

    train = pd.read_parquet(f'train.parquet')

    test = pd.read_parquet(f'test.parquet')

    specs = pd.read_parquet(f'specs.parquet')

    train_labels = pd.read_parquet(f'train_labels.parquet')

    sample_submission = pd.read_parquet(f'sample_submission.parquet')

    print(f'Reading from pkl files')

except (OSError, IOError) as e:

    train = pd.read_csv(f'{data_path}train.csv')

    test = pd.read_csv(f'{data_path}test.csv')

    specs = pd.read_csv(f'{data_path}specs.csv')

    train_labels = pd.read_csv(f'{data_path}train_labels.csv')

    sample_submission = pd.read_csv(f'{data_path}sample_submission.csv')

    

    # uncomment below when using.   

#     train.to_parquet(f'train.parquet')

#     test.to_parquet(f'test.parquet')

#     specs.to_parquet(f'specs.parquet')

#     train_labels.to_parquet(f'train_labels.parquet')

#     sample_submission.to_parquet(f'sample_submission.parquet')

    print(f'Reading from CSV files')


%%time 



train.to_parquet(f'train.parquet')

test.to_parquet(f'test.parquet')

specs.to_parquet(f'specs.parquet')

train_labels.to_parquet(f'train_labels.parquet')

sample_submission.to_parquet(f'sample_submission.parquet')

print(f'Reading from parquet files')
%%time 



train = pd.read_parquet(f'train.parquet')

test = pd.read_parquet(f'test.parquet')

specs = pd.read_parquet(f'specs.parquet')

train_labels = pd.read_parquet(f'train_labels.parquet')

sample_submission = pd.read_parquet(f'sample_submission.parquet')

print(f'Reading from pkl files')
# train = train[keep_cols]

# test = test[keep_cols]
!ls -ltrh