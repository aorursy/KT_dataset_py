"""Exploratory data analysis

    """

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# Constants

PATH = '../input/'

TRAIN = 'train.csv'

TEST = 'test.csv'



# Load and visualize data

print('Reading CSV Data...')

df_train = pd.read_csv(PATH + TRAIN)



print('Data read successfully')

df_train.head()
# Extract simple metricts from the dataframe

num_examples = len(df_train)

positive_class_ratio = round(df_train['is_duplicate'].mean()*100,2)

qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())

num_repeats = np.sum(qids.value_counts() > 1)



print('Number of examples: {}'.format(num_examples))

print('Percentage of duplicates: {}%'.format(positive_class_ratio))

print('Unique question ids: {}'.format(len(np.unique(qids))))

print('Duplicate numbers: {}'.format(num_repeats))
# Create a DataFrame of the questions indexed by qID

df_questions = pd.DataFrame({'qid': df_train['qid1'].tolist() + df_train['qid2'].tolist(),

                             'question': df_train['question1'].tolist() +

                             df_train['question2'].tolist()})



df_questions.drop_duplicates(inplace=True)

df_questions.sort_values('qid',inplace=True)



df_questions['occurrences'] = qids.value_counts().sort_index().values



df_questions.index = df_questions['qid'].values

df_questions.drop('qid',1,inplace=True)