import os

import sys

import numpy as np 

import pandas as pd

import tensorflow as tf

from transformers import BertTokenizer

sys.path.append('../input/kagglerig')

import krig

sys.path.append('../input/googlequestchallenge/kaggle-google-quest-challenge-1.0')

import googlequestchallenge as gqc
krig.seed_everything()
# Characters such as empty strings '' or numpy.inf are considered NA values

pd.set_option('use_inf_as_na', True)

pd.set_option('display.max_columns', 999)

pd.set_option('display.max_rows', 999)
IS_KAGGLE = True

#QUESTION_MODEL_NAME = 'question_bert_base_uncased_20200210_154814'

#QUESTION_MODEL_NAME = 'question_bert_large_uncased_whole_word_masking_20200210_181411'

QUESTION_MODEL_NAME = 'question_bert_base_uncased_20200210_191831'

#ANSWER_MODEL_NAME = 'answer_bert_base_uncased_20200210_160855'

#ANSWER_MODEL_NAME = 'answer_bert_large_uncased_whole_word_masking_20200210_181950'

ANSWER_MODEL_NAME = 'answer_bert_base_uncased_20200210_202316'

MAX_SEQUENCE_LENGTH = 512

STRIDE = 50

WINDOW_LENGTH = 100

QUESTION_LABELS = [

    'question_asker_intent_understanding', 'question_body_critical', 'question_conversational',

    'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer',

    'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent',

    'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice', 'question_type_compare',

    'question_type_consequence', 'question_type_definition', 'question_type_entity', 'question_type_instructions',

    'question_type_procedure', 'question_type_reason_explanation', 'question_type_spelling', 'question_well_written',

]

ANSWER_LABELS = [

    'answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance', 'answer_satisfaction',

    'answer_type_instructions', 'answer_type_procedure', 'answer_type_reason_explanation', 'answer_well_written'

]

QUESTION_FIRST_SEQUENCE = ['category', 'question_title']

ANSWER_FIRST_SEQUENCE = ['category', 'question_title']

QUESTION_SECOND_SEQUENCE = ['question_body']

ANSWER_SECOND_SEQUENCE = ['answer']

BASE_DIR = '../resources'

if IS_KAGGLE:

    BASE_DIR = '../input/googlequestchallenge/kaggle-google-quest-challenge-1.0/resources'
corrs = pd.read_csv(f'{BASE_DIR}/{QUESTION_MODEL_NAME}/corrs.csv')

print(f'q mean(corr)={corrs["corr"].mean():.4f}')

corrs.head(len(QUESTION_LABELS))
corrs = pd.read_csv(f'{BASE_DIR}/{ANSWER_MODEL_NAME}/corrs.csv')

print(f'a mean(corr)={corrs["corr"].mean():.4f}')

corrs.head(len(ANSWER_LABELS))
hist = pd.read_csv(f'{BASE_DIR}/{QUESTION_MODEL_NAME}/history.csv')

hist.head(len(hist))
hist = pd.read_csv(f'{BASE_DIR}/{ANSWER_MODEL_NAME}/history.csv')

hist.head(len(hist))
%%time

test = pd.read_csv('../input/google-quest-challenge/test.csv')

test.info()
%%time

tokenizer = BertTokenizer.from_pretrained(f'{BASE_DIR}/{QUESTION_MODEL_NAME}')

ds = gqc.Dataset(key_column='qa_id')

ds.preprocess(test, tokenizer, first_seq_columns=QUESTION_FIRST_SEQUENCE,

              second_seq_columns=QUESTION_SECOND_SEQUENCE,

            max_sequence_length=MAX_SEQUENCE_LENGTH, window_length=WINDOW_LENGTH, stride=STRIDE)

x_test = ds.inputs()

input_ids = pd.DataFrame(x_test[0])

input_ids.head(10)
%%time

path = f'{BASE_DIR}/{QUESTION_MODEL_NAME}'

model = tf.keras.models.load_model(path)

model.summary(line_length=100)
%%time

y_pred = model.predict(x_test)

print(f'y_pred.shape={np.shape(y_pred)}')
q = pd.DataFrame(y_pred, columns=QUESTION_LABELS)

q['qa_id'] = ds.df['qa_id'].values

q = q.groupby(['qa_id'], as_index=False)[QUESTION_LABELS].median()

q.info()
%%time

tokenizer = BertTokenizer.from_pretrained(f'{BASE_DIR}/{ANSWER_MODEL_NAME}')

ds = gqc.Dataset(key_column='qa_id')

ds.preprocess(test, tokenizer, first_seq_columns=ANSWER_FIRST_SEQUENCE,

              second_seq_columns=ANSWER_SECOND_SEQUENCE,

            max_sequence_length=MAX_SEQUENCE_LENGTH, window_length=WINDOW_LENGTH, stride=STRIDE)

x_test = ds.inputs()

input_ids = pd.DataFrame(x_test[0])

input_ids.head(10)
%%time

path = f'{BASE_DIR}/{ANSWER_MODEL_NAME}'

model = tf.keras.models.load_model(path)

model.summary(line_length=100)
%%time

y_pred = model.predict(x_test)

print(f'y_pred.shape={np.shape(y_pred)}')
a = pd.DataFrame(y_pred, columns=ANSWER_LABELS)

a['qa_id'] = ds.df['qa_id'].values

a = a.groupby(['qa_id'], as_index=False)[ANSWER_LABELS].median()

a.info()
qa = pd.concat([q, a], axis=1)

qa.head()
sub = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')

sub.iloc[:, 1:] = qa[QUESTION_LABELS + ANSWER_LABELS].values

gqc.check_submission(sub, shape=(476, 31), exclude={'qa_id'})

sub.head()
sub.to_csv('submission.csv', index=False)
print('\n'.join(krig.file_paths('.')))

print('\n'.join(krig.file_paths('../input')))
pd.show_versions()
!pip list