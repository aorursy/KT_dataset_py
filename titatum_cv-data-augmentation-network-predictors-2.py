import os

import datetime as dt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
input_dir = "../input"

examples_dir = os.path.join(input_dir,'cv-data-augmentation-network-predictors-1')

examples = pd.read_parquet(os.path.join(examples_dir,'positive_negative_examples.parquet.gzip'))
examples.shape
examples.sample(10)
network_predictors_2_dir = os.path.join(input_dir,'cv-feature-engineering-network-predictors-2')

print(os.listdir(network_predictors_2_dir))
network_statistics_names = ['questioners_answerers_paths', 'commenters_questioners_paths', 'commenters_answerers_paths']



for network_statistics_name in network_statistics_names:     

    print('Considering: {}'.format(network_statistics_name))



    network_statistics = pd.read_parquet(os.path.join(network_predictors_2_dir,'{}.parquet.gzip'.format(network_statistics_name)))



    print(network_statistics.shape)

    print(network_statistics.sample(3))



    examples = examples.merge(network_statistics, 

                              left_on=['emails_date', 'questions_author_id', 'answer_user_id'],

                              right_on=network_statistics.columns[[0,2,3]].values.tolist(),

                              how='left')

    examples = examples.drop(network_statistics.columns[[0,2,3]].values.tolist(), axis=1)



    examples = examples.merge(network_statistics, 

                              left_on=['emails_date', 'questions_author_id', 'answer_user_id'],

                              right_on=network_statistics.columns[[0,3,2]].values.tolist(),

                              how='left')

    examples = examples.drop(network_statistics.columns[[0,3,2]].values.tolist(), axis=1)

    examples[network_statistics.columns[1]] = examples['{}_x'.format(network_statistics.columns[1])].add(

        examples['{}_y'.format(network_statistics.columns[1])], fill_value=0) 

    examples = examples.drop(['{}_x'.format(network_statistics.columns[1]),

                              '{}_y'.format(network_statistics.columns[1])], axis=1)

    

    print('Non empty rows: {}'.format(examples[~pd.isnull(examples[network_statistics.columns[1]])].shape))

    print(examples.sample(10))

    examples[network_statistics.columns[1]] =examples[network_statistics.columns[1]].fillna(0)
examples.shape
examples.sample(10)
examples.to_parquet('positive_negative_examples.parquet.gzip', compression='gzip')
os.listdir()