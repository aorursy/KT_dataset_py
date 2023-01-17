!ls ../input
# basic

import os

import sys

from typing import Optional, List, Dict

from pathlib import Path

import json

import pickle



# data

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
datadir_name = 'tfqaconverteddata'

datadir = Path('../input/') / datadir_name

outputdir = Path('../output/')

outputdir.mkdir()

train_file = 'simplified-nq-train.jsonl'

test_file = 'simplified-nq-test.jsonl'
%%time

train_df = pd.read_csv(datadir / 'light_train.csv')

train_df.shape
%%time

test_df = pd.read_json(datadir / test_file, orient='records', lines=True)

test_df.shape
train_df.head()
question_texts = train_df['question_text'].values.tolist()
question_texts[:5]
# check length
word_lens_list = []

char_lens_list = []



for text in question_texts:

    word_lens_list.append(len(text.split()))

    char_lens_list.append(len(text))

word_lens = np.array(word_lens_list)

char_lens = np.array(char_lens_list)
word_lens.max(), word_lens.min(), word_lens.mean(), word_lens.std()
# word_lens histgram: most of questions are shorter than 15

sns.distplot(word_lens, bins=26, kde=False)
char_lens.max(), char_lens.min(), char_lens.mean(), char_lens.std()
# char_lens histgram: most sentences are around 50

sns.distplot(char_lens, bins=85, kde=False)
# some data have just 100 char length, check this

long_questions = np.array(question_texts)[char_lens == 100]

len(long_questions)
long_questions[:10]
# check words

from collections import Counter
# make all word array

words = []

for text in question_texts:

    words.extend(text.split())

all_count = Counter(words).most_common()
# see top 20

all_count[:20]
# see bottom 50, some are misspelling and special symbols and abbreviation

all_count[-50:]
# make all word array

head_words = []

for text in question_texts:

    head_words.append(text.split()[0])

head_count = Counter(head_words).most_common()
# see top 20

head_count[:20]
total_question = 307373

accm = []

counts = [c for _, c in head_count]

for i, (w, cnt) in enumerate(head_count):

    accm.append(cnt + sum(counts[:i]))

accm = np.array(accm) / np.sum(counts)
plt.title('Accumulate Ratio for word count')

plt.plot(range(len(head_count[:1000])), accm[:1000])
# top 20 get 80% of head words

list(zip(head_count[:20], accm[:20]))

# who, what, when occupy 50%, and about 70% by where, how 
# - ['who', 'who\'s', 'what', 'when', 'where', 'how', 'which', 'why', 'what\'s'] or not
train_df['annotations'].iloc[0]
# one record is a str of list of json dict

# there are keys:

keys = ['yes_no_answer', 'long_answer', 'short_answers', 'annotation_id']

# long_answer has keys:

long_answer_keys = ['start_token', 'candidate_index', 'end_token']

# short_answers is a list of dict and keys:

short_answers_keys = ['start_token', 'end_token']
# make DataFrame for annotation records

def annotations2value(x: str, key: str) -> Optional[str]:

    if isinstance(x, str):

        x = x.replace('\'', '"')

    x = json.loads(x)

    if isinstance(x, list):

        assert len(x) == 1, f'{x} has multi answers'

        x = x[0]

    v = x[key] if key in x else None

    return v





def short2value(x: List[Dict[str, int]], key: str) -> Optional[List[int]]:

    if len(x) == 0:

        return ''

    return ','.join([str(xx[key]) for xx in x])



for k in keys:

    train_df[k] = train_df['annotations'].apply(lambda x: annotations2value(x, k))

    

for k in long_answer_keys:

    train_df['long_' + k] = train_df['long_answer'].apply(lambda x: x[k])

    

for k in short_answers_keys:

    train_df['short_' + k] = train_df['short_answers'].apply(lambda x: short2value(x, k))

    

def get_short_count(x: str):

    if x == '':

        return 0

    x = x.split(',')

    return len(x)



train_df['short_answer_count'] = train_df['short_start_token'].apply(get_short_count)
# del train_df['annotations'], train_df['long_answer'], train_df['short_answers']

train_df.head()

# there are some some values -1 in long
# save annotation DataFrame

# train_df.to_csv(outputdir / 'annotations_detail.csv', index=False)

annotation_df = train_df

del train_df
annotation_df = pd.read_csv(outputdir / 'annotations_detail.csv')

annotation_df.shape
# check yes_no_answer

sns.countplot(x=annotation_df['yes_no_answer'])
# most of answer is None, check only yes_no

sns.countplot(x=annotation_df[annotation_df['yes_no_answer'] != 'NONE']['yes_no_answer'])
# next check the number of short answer

print(annotation_df['short_answer_count'].describe())

sns.countplot(x=annotation_df['short_answer_count'])

# short answer has 25 answers at max but most of them are 1 or 2, or 0.

# 2/3 of short answers are None.

# what means for multiple short answer? check it later.
# what value is in short_start/end when answer is yes/no?

sns.countplot(x=annotation_df[annotation_df['yes_no_answer'] != 'NONE']['short_answer_count'])

# it refer to zero (None). if the answer is yes/no, there are no short_answer by token index.
# how many records are zero short answer?

annotation_df.query("yes_no_answer == 'NONE' and short_answer_count == 0")['example_id'].count()
# check length of short answer
# check short_start_token if or not it has anomaly
short_start_tokens = []

short_end_tokens = []

short_token_lengths = []

for start_token, end_token in annotation_df[['short_start_token', 'short_end_token']].values:

    if start_token is np.nan and end_token is np.nan:

        continue

    short_start_tokens.extend([int(t) for t in start_token.split(',')])

    short_end_tokens.extend([int(t) for t in end_token.split(',')])

short_start_tokens = np.array(short_start_tokens)

short_end_tokens = np.array(short_end_tokens)

short_token_lengths = short_end_tokens - short_start_tokens
short_df = pd.DataFrame(

    zip(short_start_tokens, short_end_tokens, short_token_lengths),

    columns=['short_start_tokens', 'short_end_tokens', 'short_token_lengths']

)

print(short_df['short_start_tokens'].describe())

print(short_df['short_end_tokens'].describe())

print(short_df['short_token_lengths'].describe())

sns.countplot(x=short_df[short_df['short_token_lengths'] < 20]['short_token_lengths'])
# max length of short answer is 250, is it anomaly value?

print('more than 20:', short_df[short_df['short_token_lengths'] >= 20]['short_token_lengths'].count())

print('more than 50:', short_df[short_df['short_token_lengths'] >= 50]['short_token_lengths'].count())

print('more than 100:', short_df[short_df['short_token_lengths'] >= 100]['short_token_lengths'].count())

print('more than 150:', short_df[short_df['short_token_lengths'] >= 150]['short_token_lengths'].count())

print('more than 200:', short_df[short_df['short_token_lengths'] >= 200]['short_token_lengths'].count())

print('more than 220:', short_df[short_df['short_token_lengths'] >= 220]['short_token_lengths'].count())

# There are 35 samples whose length are longer than 100, what is short

# check this not short short answer later
# multiple short answer is same phrase?

# check length of each multiple short answer

not_same_count = 0

for start_token, end_token in annotation_df[['short_start_token', 'short_end_token']].values:

    if start_token is np.nan and end_token is np.nan:

        continue

    starts = np.array([int(t) for t in start_token.split(',')])

    ends = np.array([int(t) for t in end_token.split(',')])

    if len(starts) <= 1:

        continue

    diff = ends - starts

    is_all_same = (diff == diff[0]).all()

    if not is_all_same:

        not_same_count += 1



print(not_same_count)



# almost multiple answers are not same, I have to check the text later.
# long answer

annotation_df = pd.read_csv(outputdir / 'annotations_detail.csv')

annotation_df.head()
# long answers have candidates and one of them are correct

# addition to that, it has 'top_level' column which indicate the answer is included in another answer.

annotation_df['long_answer_candidates'].iloc[0]
# firstly, check the number of candidates

la_keys = ['start_token', 'end_token', 'top_level']

def la2value(key, x):

    if isinstance(x, str):

        x = x.replace('\'', '"')

        x = x.replace('True', '1')

        x = x.replace('False', '0')

    x = json.loads(x)

    values = []

    for candidate in x:

        values.append(int(candidate[key]))

    values = ','.join([str(v) for v in values])

    return values

for key in la_keys:

    print('start:', key)

    annotation_df['long_' + key + 's'] = annotation_df['long_answer_candidates'].apply(lambda x: la2value(key, x))
annotation_df.to_csv(outputdir / 'annotations_detail.csv', index=False)

annotation_df.head()
# firstly check long answer candidates num

annotation_df['long_candidates_num'] = annotation_df['long_start_tokens'].apply(lambda x: len(x.split(',')))

print(annotation_df['long_candidates_num'].describe())

_ = plt.figure(figsize=(20, 10))

# sns.countplot(x=annotation_df['long_candidates_num'])

sns.countplot(x=annotation_df[annotation_df['long_candidates_num'] < 150]['long_candidates_num'])
# check the record that has no long answer. token == -1 means no long answer

annotation_df['no_long_answer'] = annotation_df['long_start_token'] == -1

print(annotation_df['no_long_answer'].describe())

sns.countplot(x=annotation_df['no_long_answer'])

# half of data have no long answer...
# in case there is no long answer, there shoud be no short answer. check this.

annotation_df[annotation_df['no_long_answer']]['short_answer_count'].describe()
# so then, there are

# - 307373 record

# - 196649 have no short answer

# - 155225 have no long answer (off course no short answer if no long answer)
# check top_level

annotation_df['long_top_level_1_num'] = annotation_df['long_top_levels'].apply(lambda x: sum([int(xx) for xx in x.split(',')]))

annotation_df['long_top_level_0_num'] = annotation_df['long_top_levels'].apply(lambda x: len(x.split(',')) - sum([int(xx) for xx in x.split(',')]))
_ = plt.figure(figsize=(18, 4))

# sns.countplot(x=annotation_df['long_top_level_0_num'])

sns.countplot(x=annotation_df[annotation_df['long_top_level_0_num'] <= 100]['long_top_level_0_num'])

_ = plt.figure(figsize=(18, 4))

# sns.countplot(x=annotation_df['long_top_level_1_num'])

sns.countplot(x=annotation_df[annotation_df['long_top_level_1_num'] <= 100]['long_top_level_1_num'])
# by histgram

sns.distplot(annotation_df[annotation_df['long_top_level_0_num'] <= 100]['long_top_level_0_num'], bins=100)

sns.distplot(annotation_df[annotation_df['long_top_level_1_num'] <= 100]['long_top_level_1_num'], bins=100)
num0 = annotation_df['long_top_level_0_num'].sum()

num1 = annotation_df['long_top_level_1_num'].sum()

sns.barplot(y=[num0, num1], x=['0', '1'])

# about 2/3 data are top_level = 0
# annotation_df['long_correct_top_level']

long_correct_top_level = np.zeros(annotation_df.values.shape[0])

for i, target in enumerate(annotation_df['long_candidate_index'].values):

    long_correct_top_level[i] = annotation_df['long_top_levels'].iloc[i].split(',')[target]

annotation_df['long_correct_top_level'] = long_correct_top_level
sns.countplot(x=annotation_df['long_correct_top_level'])

# most of data are top_level = 1 for only correct data

# even though 2/3 data are top_level = 0 plotted above
# next, check long answer length

annotation_df['long_correct_length'] = annotation_df['long_end_token'] - annotation_df['long_start_token']
print(annotation_df[annotation_df['long_correct_length'] >= 1]['long_correct_length'].describe())

_ = plt.figure(figsize=(20, 6))

_ = sns.distplot(annotation_df.query('long_correct_length > 1 & long_correct_length < 1000')['long_correct_length'], bins=100)
# all long candidates

long_start_tokens = []

long_end_tokens = []

long_token_lengths = []

for start_token, end_token in annotation_df[['long_start_tokens', 'long_end_tokens']].values:

    long_start_tokens.extend([int(t) for t in start_token.split(',')])

    long_end_tokens.extend([int(t) for t in end_token.split(',')])

long_start_tokens = np.array(long_start_tokens)

long_end_tokens = np.array(long_end_tokens)

long_token_lengths = long_end_tokens - long_start_tokens
long_df = pd.DataFrame(

    long_token_lengths,

    columns=['long_token_lengths']

)

del long_start_tokens, long_end_tokens, long_token_lengths
# print(long_df['long_start_tokens'].describe())

# print(long_df['long_end_tokens'].describe())

print(long_df['long_token_lengths'].describe())
_ = plt.figure(figsize=(20, 6))

_ = sns.distplot(

    long_df.query('long_token_lengths < 1000')['long_token_lengths'],

    bins=100,

    label='all_long_candidate_answer'

)

_ = sns.distplot(

    annotation_df.query('long_correct_length > 1 & long_correct_length < 1000')['long_correct_length'],

    bins=100,

    label='correct_long_answer'

)

_ = plt.legend()

# big difference in length distribution of candidates and correct answer 
annotation_df.to_csv(outputdir / 'annotations_detail.csv', index=False)
# todo

# - multiple short answer text

# - not short short answer

# - answer check by question type