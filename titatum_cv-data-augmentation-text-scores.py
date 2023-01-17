import os

import datetime as dt

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
input_dir = "../input"

examples_dir = os.path.join(input_dir,'cv-data-augmentation-network-predictors-2')

examples = pd.read_parquet(os.path.join(examples_dir,'positive_negative_examples.parquet.gzip'))
examples.shape
examples.sample(10)
def get_lsi_score(row, questions_topics, user_topics):

    if ((row['questions_id'] in questions_topics.index) &

        (row['answer_user_id'] in user_topics.index)):

        lsi_1 = questions_topics.loc[row['questions_id']].values

        lsi_2 = user_topics.loc[row['answer_user_id']].values

        return np.dot(lsi_1,lsi_2) / np.sqrt(np.dot(lsi_1,lsi_1) * np.dot(lsi_2,lsi_2))

    else:

        return 0.0
lsi_predictors_dir = os.path.join(input_dir,'cv-feature-engineering-text-scores')

vocabulary_size = 10000

num_topics = 50
questions_topics = pd.read_parquet(

    os.path.join(lsi_predictors_dir,

                 'questions_topics_vs_{}_nt_{}.parquet.gzip'.format(vocabulary_size, num_topics)))

questions_topics.head(2)
merged_user_topics = pd.read_parquet(

    os.path.join(lsi_predictors_dir,

                 'merged_user_topics_vs_{}_nt_{}.parquet.gzip'.format(vocabulary_size, num_topics)))

merged_user_topics.head(2)
%%time

examples['LSI_Score'] = examples.apply(get_lsi_score, axis=1, 

                                       questions_topics=questions_topics,

                                       user_topics=merged_user_topics)

examples['LSI_Score'] = examples['LSI_Score'].fillna(0)
examples['LSI_Score'].describe()
examples.to_parquet('positive_negative_examples.parquet.gzip', compression='gzip')
os.listdir()