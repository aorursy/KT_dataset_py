import numpy as np

import pandas as pd



# Any results you write to the current directory are saved as output.

import os

import gc

import matplotlib.pyplot as plt

import json



from collections import Counter
# Input data files are available in the "../input/" directory.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = '/kaggle/input/tensorflow2-question-answering/'

train_path = 'simplified-nq-train.jsonl'

test_path = 'simplified-nq-test.jsonl'

sample_submission_path = 'sample_submission.csv'
train_line_count = 0

with open(path+train_path) as f:

    for line in f:

        line = json.loads(line)

        train_line_count += 1

train_line_count
def read_data(path, sample = True, chunksize = 30000):

    if sample == True:

        df = []

        with open(path, 'rt') as reader:

            for i in range(chunksize):

                df.append(json.loads(reader.readline()))

        df = pd.DataFrame(df)

    else:

        df = pd.read_json(path, orient = 'records', lines = True)

        gc.collect()

    return df
train = read_data(path+train_path, sample = True, chunksize=100000)

print("train shape", train.shape)

train[:10]
test = read_data(path+test_path, sample = False)

print("test shape", test.shape)

test[:10]
sample_submission = pd.read_csv(path + sample_submission_path)

print("Sample submission shape", sample_submission.shape)
sample_submission[:10]
def missing_values(df):

    df = pd.DataFrame(df.isnull().sum()).reset_index()

    df.columns = ['features', 'n_missing_values']

    return df
missing_values(train)
missing_values(test)
train.columns
# Question text

train.loc[0, 'question_text']
train.loc[0, 'document_text']
train.loc[0, 'document_text'].split()[:100]
train.loc[0, 'long_answer_candidates'][:10]
train['annotations'][:10]
# Make a dataframe to accumulate answer types

answer_num_annotations = train.annotations.apply(lambda x: len(x))

Counter(answer_num_annotations)
train['annotations'][0]
set().union(*(d[0].keys() for d in train['annotations']))
answer_summary = train['example_id'].to_frame()
answer_summary['annotation_id'] = train.annotations.apply(lambda x: x[0]['annotation_id'])
answer_summary[:10]
answer_yes_no = train.annotations.apply(lambda x: x[0]['yes_no_answer'])

yes_no_answer_counts = Counter(answer_yes_no)

yes_no_answer_counts
ks = [k for k in yes_no_answer_counts.keys()]

vs = [yes_no_answer_counts[k] for k in ks]



plt.bar(ks, vs)
percent_yes_no = 1 - yes_no_answer_counts['NONE'] / sum(yes_no_answer_counts.values())

print(percent_yes_no, "of the questions have yes/no answers given")
answer_yes_no_cleaned = answer_yes_no.apply(lambda x: None if x == 'NONE' else x)

answer_summary['has_yes_no'] = answer_yes_no_cleaned.apply(lambda x: x is not None)

answer_summary['yes_no'] = answer_yes_no_cleaned
train.annotations.apply(lambda x: x[0]['short_answers'])[:10]
answer_short = train.annotations.apply(lambda x: [(y['start_token'], y['end_token']) for y in x[0]['short_answers']])

num_short_answers = answer_short.apply(lambda x: len(x))

short_answer_counts = Counter(num_short_answers)

short_answer_counts
ks = [k for k in short_answer_counts.keys()]

ks.sort()

vs = [short_answer_counts[k] for k in ks]



plt.bar(ks, vs)
percent_short = 1 - short_answer_counts[0] / sum(short_answer_counts.values())

print(percent_short, "of the questions have at least one short answer")
answer_summary['has_short_answers'] = num_short_answers.apply(lambda x: x>0)

answer_summary['num_short_answers'] = num_short_answers

answer_summary['answer_short'] = answer_short.apply(lambda x: x if len(x) > 0 else None)
train.annotations.apply(lambda x: x[0]['long_answer'])[:20]
train.loc[0, 'annotations'][0]['long_answer']
answer_long = train.annotations.apply(lambda x: (x[0]['long_answer']['start_token'], x[0]['long_answer']['end_token']))

answer_long_cleaned = answer_long.apply(lambda x: x if x != (-1, -1) else None)

num_long_answers = answer_long_cleaned.apply(lambda x: 1 if x else 0)

long_answer_counts = Counter(num_long_answers)

long_answer_counts
ks = [k for k in long_answer_counts.keys()]

ks.sort()

vs = [long_answer_counts[k] for k in ks]



plt.bar(ks, vs)
percent_long = 1 - long_answer_counts[0] / sum(long_answer_counts.values())

print (percent_long, "of the questions have at least one long answer")
answer_summary['has_long_answer'] = num_long_answers.apply(lambda x: x>0)

answer_summary['num_long_answers'] = num_long_answers

answer_summary['answer_long'] = answer_long_cleaned
candidate_indices = train.annotations.apply(lambda x: (x[0]['long_answer']['candidate_index']))
answer_summary['long_candidate_index'] = candidate_indices
answer_summary[:10]
summary = answer_summary.apply(lambda row: 

                               True if (row['has_yes_no'] or row['has_short_answers'] or row['has_long_answer'])

                               else False, axis=1)

summary[:10]
Counter(summary)
answer_summary["summary"] = summary
answer_summary.groupby(['has_yes_no', 'has_short_answers', 'has_long_answer']).size().reset_index()
answer_summary[:10]