import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder



import tensorflow as tf



import json

import gc

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = '/kaggle/input/tensorflow2-question-answering/'

train_path = 'simplified-nq-train.jsonl'

test_path = 'simplified-nq-test.jsonl'

sample_submission_path = 'sample_submission.csv'
def read_data(path, sample = True, chunksize = 30000):

    if sample == True:

        df = []

        with open(path, 'rt') as reader:

            for i in range(chunksize):

                df.append(json.loads(reader.readline()))

        df = pd.DataFrame(df)

        print('Our sampled dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))

    else:

        df = pd.read_json(path, orient = 'records', lines = True)

        print('Our dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))

        gc.collect()

    return df
train = read_data(path+train_path, sample = True)

test = read_data(path+test_path, sample = False)

train.head()
test.head()
sample_submission = pd.read_csv(path + sample_submission_path)

print('Our sample submission have {} rows'.format(sample_submission.shape[0]))

sample_submission.head()
def missing_values(df):

    df = pd.DataFrame(df.isnull().sum()).reset_index()

    df.columns = ['features', 'n_missing_values']

    return df

missing_values(train)
missing_values(test)
question_text_0 = train.loc[0, 'question_text']

question_text_0
document_text_0 = train.loc[0, 'document_text'].split()

" ".join(document_text_0[:100])
from IPython.display import HTML,display

display(HTML(" ".join(document_text_0)))
long_answer_candidates_0 = train.loc[0, 'long_answer_candidates']

long_answer_candidates_0[0:20]
annotations_0 = train.loc[0, 'annotations']

annotations_0