# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#!git clone https://github.com/dwyl/english-words  #english words dictionary
import pandas as pd

import json

from tqdm import tqdm
df = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')
# Getting the question_title, question_body and qustion_answer for vectorizing

df_questions_answer_title = []

for i in range(len(df['question_body'])):

    df_questions_answer_title.append(df['question_title'][i]+' '+df['question_body'][i]+' '+df['answer'][i])
import re

def eliminate_non_char(sample):

    regex = re.compile("""[,\.!?'"123456789]""")

    sample = regex.sub('',sample )

    sample = re.sub(r'\([^)]*\)', '', sample)

    sample = sample.replace('\n','')

    new_sample = []

    for word in sample.split():

            new_sample.append(word)

    return new_sample



    

#samples =[eliminate_non_english_words(x.lower()) for x in df_questions_answer_title] # question body + question answer + title

samples =[eliminate_non_char(x.lower()) for x in df_questions_answer_title] # question body + question answer + title
labels = df
from sklearn.model_selection import train_test_split

X_train, X_dev, y_train, y_dev = train_test_split(samples, labels, test_size=0.20)
#y_train = np.array(y_train)

#y_test = np.array(y_test)
from gensim.models import Word2Vec

w2v_size = 300

w2v_model = Word2Vec(min_count=50,

                     window=4,

                     size=w2v_size,

                     workers=2)

w2v_model.build_vocab(X_train)

w2v_model.train(X_train, total_examples=w2v_model.corpus_count, epochs=30)
# vectorizing train data

padding= np.zeros(w2v_size)

X_train_vec = []

for sample in X_train:

    vec = []

    k = 0

    for word in sample:

        if k >= 50:

            break

        k += 1

        try:

            vec.append(w2v_model.wv[word])

        except:

            vec.append(padding)

    while k < 50:

        vec.append(padding)

        k += 1

    X_train_vec.append(np.array(vec))

X_train_vec = np.array(X_train_vec)



# vectorizing dev data

padding= np.zeros(w2v_size)

X_dev_vec = []

for sample in X_dev:

    vec = []

    k = 0

    for word in sample:

        if k >= 50:

            break

        k += 1

        try:

            vec.append(w2v_model.wv[word])

        except:

            vec.append(padding)

    while k < 50:

        vec.append(padding)

        k += 1

    X_dev_vec.append(np.array(vec))

X_dev_vec = np.array(X_dev_vec)
X_dev = X_dev_vec

X_train = X_train_vec

print(X_train.shape)

print(X_dev.shape)

print(y_train.shape)

print(y_dev.shape)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, InputLayer, LSTM, Bidirectional, GlobalMaxPooling1D, Conv1D, Dropout, MaxPool1D

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback, EarlyStopping
def build_model():

    model = Sequential()

    model.add(LSTM(32, return_sequences = 'True'))

    model.add(Dropout(0.5))

    model.add(LSTM(16))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model



def train_model(model, X_train, y_train, X_test, y_test):

    mcp_save = ModelCheckpoint('model.hdf5', save_best_only=True, monitor='val_loss', mode='min')

    model.fit(X_train, y_train,validation_data=[X_test, y_test], batch_size=32, epochs=100,  

                        callbacks= [

                              EarlyStopping(patience=10, monitor='val_loss', mode='min'),

                              mcp_save,

                              ReduceLROnPlateau(factor=.3)

                         ])

    model.load_weights(filepath = 'model.hdf5')

    return model
# m = build_model()

# train_model(m, X_train, np.array(y_train['question_well_written']), X_dev, np.array(y_dev['question_well_written']))
classes =['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']
test_data = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

sample_subbmision = pd.read_csv('/kaggle/input/google-quest-challenge/sample_submission.csv')
df_questions_answer_title = []

for i in range(len(test_data['question_body'])):

    df_questions_answer_title.append(df['question_title'][i]+' '+df['question_body'][i]+' '+df['answer'][i])

print(len(df_questions_answer_title))

samples =[eliminate_non_char(x.lower()) for x in df_questions_answer_title] # question body + question answer + title

X_test = samples

# vectorizing test data

padding= np.zeros(w2v_size)

X_test_vec = []

for sample in X_test:

    vec = []

    k = 0

    for word in sample:

        if k >= 50:

            break

        k += 1

        try:

            vec.append(w2v_model.wv[word])

        except:

            vec.append(padding)

    while k < 50:

        vec.append(padding)

        k += 1

    X_test_vec.append(np.array(vec))

X_test_vec = np.array(X_test_vec)

X_test = X_test_vec

print(X_test.shape)
#subbmision = pd.DataFrame(columns = ['qa_id']+classes) 
predictions = {}

k = 1

for cl in classes:

    m = build_model()

    print('Training for '+cl+'\n')

    m = train_model(m, X_train, np.array(y_train[cl]), X_dev, np.array(y_dev[cl]))

    pred = m.predict(X_test)

    predictions[cl] = pred
df_columns  = ['qa_id']

i = 1

sub_df = pd.DataFrame(columns = df_columns)

for cl in classes:

    pred = []

    for el in predictions[cl]:

        #pred.append(el[0])

        pred.append("{:.5f}".format(el[0]))

    sub_df.insert(i, cl, pred, True)

    i += 1
sub_df['qa_id'] = test_data['qa_id']
sub_df.to_csv("submission.csv", index=False)