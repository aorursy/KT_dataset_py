import os

import datetime as dt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
input_dir = "../input"

examples_dir = os.path.join(input_dir,'cv-data-augmentation-activity-predictors')

examples = pd.read_parquet(os.path.join(examples_dir,'positive_negative_examples.parquet.gzip'))
examples.shape
examples.sample(10)
network_predictors_1_dir = os.path.join(input_dir,'cv-feature-engineering-network-predictors-1')

print(os.listdir(network_predictors_1_dir))
network_statistics_map = {'users_users_shared_schools': ['questions_author_id', 'answer_user_id'], 

                          'users_users_shared_tags': ['questions_author_id', 'answer_user_id'], 

                          'users_users_shared_groups': ['questions_author_id', 'answer_user_id'],

                          'questions_users_shared_tags': ['questions_id', 'answer_user_id']}



for network_statistics_name in network_statistics_map.keys():

    print('Considering: {}'.format(network_statistics_name))

    

    network_statistics = pd.read_parquet(os.path.join(network_predictors_1_dir,'{}.parquet.gzip'.format(network_statistics_name)))



    print(network_statistics.shape)

    print(network_statistics.head(3))



    examples = examples.merge(network_statistics, 

                                  on=network_statistics_map[network_statistics_name],

                                  how='left')



    print('Non empty rows: {}'.format(examples[~pd.isnull(examples[network_statistics.columns[-1]])].shape))

    print(examples.sample(10))

    examples[network_statistics.columns[-1]] = examples[network_statistics.columns[-1]].fillna(0)
examples.shape
examples.sample(10)
examples.to_parquet('positive_negative_examples.parquet.gzip', compression='gzip')
os.listdir()