import re

import os

import gc

import json

import pickle

import fasttext

import Levenshtein

import numpy as np 

import pandas as pd

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.metrics import accuracy_score, f1_score

from tqdm import tqdm_notebook as tqdm 

from Levenshtein import ratio as levenshtein_distance

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction import text

from scipy import spatial

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as patches

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go

import plotly.express as px

pd.set_option('max_columns', 1000)

from bokeh.models import Panel, Tabs

from bokeh.io import output_notebook, show

from bokeh.plotting import figure

import lightgbm as lgb

import plotly.figure_factory as ff

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing import text, sequence

from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm_notebook as tqdm

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Masking

from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, Dropout

from tensorflow.keras.preprocessing import text, sequence

from multiprocessing import Pool

print (" Data is imported to squeeze and play")

def build_train(train_path, n_rows=200000, sampling_rate=15):

    with open(train_path) as f:

        processed_rows = []



        for i in tqdm(range(n_rows)):

            line = f.readline()

            if not line:

                break



            line = json.loads(line)



            text = line['document_text'].split(' ')

            question = line['question_text']

            annotations = line['annotations'][0]



            for i, candidate in enumerate(line['long_answer_candidates']):

                label = i == annotations['long_answer']['candidate_index']



                start = candidate['start_token']

                end = candidate['end_token']



                if label or (i % sampling_rate == 0):

                    processed_rows.append({

                        'text': " ".join(text[start:end]),

                        'is_long_answer': label,

                        'question': question,

                        'annotation_id': annotations['annotation_id']

                    })



        train = pd.DataFrame(processed_rows)

        

        return train

def build_test(test_path):

    with open(test_path) as f:

        processed_rows = []



        for line in tqdm(f):

            line = json.loads(line)



            text = line['document_text'].split(' ')

            question = line['question_text']

            example_id = line['example_id']



            for candidate in line['long_answer_candidates']:

                start = candidate['start_token']

                end = candidate['end_token']



                processed_rows.append({

                    'text': " ".join(text[start:end]),

                    'question': question,

                    'example_id': example_id,

                    'sequence': f'{start}:{end}'



                })



        test = pd.DataFrame(processed_rows)

    

    return test
directory = '/kaggle/input/tensorflow2-question-answering/'
train_path = directory + 'simplified-nq-train.jsonl'

test_path = directory + 'simplified-nq-test.jsonl'



train = build_train(train_path)

test = build_test(test_path)
train.head()
test.head()
def compute_text_and_questions(train, test, tokenizer):

    train_text = tokenizer.texts_to_sequences(train.text.values)

    train_questions = tokenizer.texts_to_sequences(train.question.values)

    test_text = tokenizer.texts_to_sequences(test.text.values)

    test_questions = tokenizer.texts_to_sequences(test.question.values)

    

    train_text = sequence.pad_sequences(train_text, maxlen=300)

    train_questions = sequence.pad_sequences(train_questions)

    test_text = sequence.pad_sequences(test_text, maxlen=300)

    test_questions = sequence.pad_sequences(test_questions)

    

    return train_text, train_questions, test_text, test_questions
tokenizer = text.Tokenizer(lower=False, num_words=80000)



for text in tqdm([train.text, test.text, train.question, test.question]):

    tokenizer.fit_on_texts(text.values)
train_target = train.is_long_answer.astype(int).values
train_text, train_questions, test_text, test_questions = compute_text_and_questions(train, test, tokenizer)

del train
def build_embedding_matrix(tokenizer, path):

    embedding_matrix = np.zeros((tokenizer.num_words + 1, 300))

    ft_model = fasttext.load_model(path)



    for word, i in tokenizer.word_index.items():

        if i >= tokenizer.num_words - 1:

            break

        embedding_matrix[i] = ft_model.get_word_vector(word)

    

    return embedding_matrix
def build_model(embedding_matrix):

    embedding = Embedding(

        *embedding_matrix.shape, 

        weights=[embedding_matrix], 

        trainable=False, 

        mask_zero=True

    )

    

    q_in = Input(shape=(None,))

    q = embedding(q_in)

    q = SpatialDropout1D(0.2)(q)

    q = Bidirectional(LSTM(100, return_sequences=True))(q)

    q = GlobalMaxPooling1D()(q)

    

    

    t_in = Input(shape=(None,))

    t = embedding(t_in)

    t = SpatialDropout1D(0.2)(t)

    t = Bidirectional(LSTM(150, return_sequences=True))(t)

    t = GlobalMaxPooling1D()(t)

    

    hidden = concatenate([q, t])

    hidden = Dense(300, activation='relu')(hidden)

    hidden = Dropout(0.5)(hidden)

    hidden = Dense(300, activation='relu')(hidden)

    hidden = Dropout(0.5)(hidden)

    

    out1 = Dense(1, activation='sigmoid')(hidden)

    

    model = Model(inputs=[t_in, q_in], outputs=out1)

    model.compile(loss='binary_crossentropy', optimizer='adam')



    return model
from pathlib import Path

PATH = Path('/kaggle/input/tensorflow2-question-answering')

!ls -1 {PATH}

TEST_TOTAL = 346
def get_joined_tokens(answer: dict) -> str:

    return '%d:%d' % (answer['start_token'], answer['end_token'])



def get_pred(json_data: dict) -> dict:

    ret = {'short': 'YES', 'long': ''}

    candidates = json_data['long_answer_candidates']

    

    paragraphs = []

    tokens = json_data['document_text'].split(' ')

    for cand in candidates:

        start_token = tokens[cand['start_token']]

        if start_token == '<P>' and cand['top_level'] and cand['end_token']-cand['start_token']>35:

            break

    else:

        cand = candidates[0]

        

    ret['long'] = get_joined_tokens(cand)

    

    id_ = str(json_data['example_id'])

    ret = {id_+'_'+k: v for k, v in ret.items()} 

    return ret



preds = dict()



with open(PATH / 'simplified-nq-test.jsonl', 'r') as f:

    for line in tqdm(f, total=TEST_TOTAL):

        json_data = json.loads(line) 

        prediction = get_pred(json_data)

        preds.update(prediction)

            

submission = pd.read_csv(PATH / 'sample_submission.csv')

submission['PredictionString'] = submission['example_id'].map(lambda x: preds[x])

submission.to_csv('submission.csv', index=False)
