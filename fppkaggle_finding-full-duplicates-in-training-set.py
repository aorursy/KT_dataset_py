import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import string

from copy import deepcopy



pd.set_option('expand_frame_repr', False)

pd.set_option('display.max_colwidth', -1)



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

# samplesub_df = pd.read_csv("../input/sample_submission.csv')



print('Train shape', train_df.shape)

print('Test shape', test_df.shape)

train_full_duplies = train_df[train_df['question1'] == train_df['question2']]

print('Fully equal training questions - no processing: ', train_full_duplies.shape[0])



train_drop = train_df.dropna(how="any")

# train_df[~train_df['id'].isin(train_drop['id'])]

train_nan = train_df[train_df['id'].isin(train_drop['id']) == False]

print('Training questions with NaN - no processing: ', train_nan.shape[0])

train_no_nan = train_drop

train_no_nan_raw = deepcopy(train_drop) # we need this later

train_drop = None



for index, row in train_nan.iterrows():

    print('Train_NaN: ', row['id'], row['qid1'], row['qid2'], ' Q1: ', row['question1'], ' Q2: ', row['question2'],

          ' is duplicate: ', row['is_duplicate'])

def remove_punct(val):

    # remove all punctuation chars

    regex = re.compile('[%s]' % re.escape(string.punctuation))

    sentence = regex.sub('', val).lower()

    

    return sentence



def clean_dataframe(data):

    # first remove punctuation than make lowercase

    for col in ['question1', 'question2']:

        data[col] = data[col].apply(remove_punct)



    return data



train_data_clean = clean_dataframe(train_no_nan)

# print(train_data_clean.head(5))



train_full_duplies_punct = train_data_clean[train_data_clean['question1'] == train_data_clean['question2']]

print('Fully equal training questions punctuation removed: ', train_full_duplies_punct.shape[0])

i_wrong_class = 0

for index, row in train_full_duplies_punct.iterrows():

    if row['is_duplicate'] == 0:

        i_wrong_class += 1

        print('Train-duplicates potentially wrongly classified: ', row['id'], row['qid1'], row['qid2'], ' [Q1]: ', row['question1'], ' [Q2]: ',

              row['question2'], ' [is duplicate]: ', row['is_duplicate'], '\n')

        raw_row = train_no_nan_raw.loc[(train_no_nan_raw['qid1'] == row['qid1']) & (train_no_nan_raw['is_duplicate'] == 0)]

        print('Orig: [raw Q1]:', format(raw_row['question1']), ' [raw Q2]: ', format(raw_row['question2']), '\n' )



print ('Total of potentially wrongly classified in training set: ', format(i_wrong_class))