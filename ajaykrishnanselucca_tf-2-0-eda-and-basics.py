# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import json

import gc

import sys



from tensorflow import keras

from tensorflow.keras import layers



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
bs_path = '/kaggle/input/tensorflow2-question-answering/'

train_file = 'simplified-nq-train.jsonl'

test_file = 'simplified-nq-test.jsonl'
def read_data(file_name, num_records = sys.maxsize): # = sys.maxsize

    current_record = 1

    records = []

    

    with open(os.path.join(bs_path, file_name)) as file:

        line = file.readline()

        while(line):

            records.append(json.loads(line))

            line = file.readline()

            if current_record > num_records:

                break

                

            if current_record % 5000 == 0:

                print(current_record)

                gc.collect()

                

            current_record = current_record + 1

    df = pd.DataFrame(records)

    return df
%%time

max_records = 5000

df_train = read_data(train_file, max_records)

gc.collect()
df_train.head()
df_train['question_text'][0]




df_train['document_text'][0:3]







df_train['long_answer_candidates'][0][54]







df_train['document_text'][0]







df_train['annotations'][0]



df_train['yes_no_answer'] = [item[0]['yes_no_answer'] for item in df_train['annotations']]

df_train['long_answer'] = [item[0]['long_answer'] for item in df_train['annotations']]

df_train['short_answers'] = [item[0]['short_answers'] for item in df_train['annotations']]

df_train['annotation_id'] = [item[0]['annotation_id'] for item in df_train['annotations']]
df_train['yes_no_answer'].value_counts()
## Getting values out of annotations
#Short answer

start_vals = []

end_vals = []



for item in df_train['short_answers']:

    start = -1

    end = -1

    if len(item) > 0:

        start = item[0]['start_token']

        end = item[0]['end_token']

    #if len(item) > 1: #TODO -> there are cases with more than one correct long/short answers, handle/check it

    #    print(item)

    start_vals.append(start)

    end_vals.append(end)

df_train['short_answer_start'] = start_vals

df_train['short_answer_end'] = end_vals



# del df_train['short_answers'] #TODO
#Long answer

    

df_train['long_answer_start'] = [item['start_token'] for item in df_train['long_answer']]

df_train['long_answer_end'] = [item['end_token'] for item in df_train['long_answer']]

df_train['long_answer_index'] = [item['candidate_index'] for item in df_train['long_answer']]



# del df_train['long_answer'] #TODO




df_train.head()







df_train.isnull().sum()







df_train.head()



df_train = df_train[['document_text', 'question_text', 'short_answer_start', 'short_answer_end']]




df_train.head(1)



df_train['document_text'][0].split()[1955:1969]




df_train['document_text'][0].split()



## MODEL ARCHITECTURE



#inputs = [['wikipedia', 'dcoument', 'text', '...'], ['question', 'text', '...']]

inputs = [['wikipedia', 'dcoument', 'text', '...']]

outputs = [0,       1, 1, '...']




doc_text = df_train['document_text'][0]

short_answer_start = df_train['short_answer_start'][0]

short_answer_end = df_train['short_answer_end'][0]



que_text = df_train['question_text'][0]



from keras.preprocessing.sequence import pad_sequences



max_len_document = 2500

max_len_que = 50

max_len_word = 40

max_len_input = max_len_document + max_len_que



doc_text_lst = []

doc_text_lst.append(doc_text.split())



que_text_lst = []

que_text_lst.append(que_text.split())



output_label = np.zeros(max_len_document)



doc_text_lst = pad_sequences(doc_text_lst, maxlen=max_len_document, dtype=object, padding='post', truncating='post', value='')



que_text_lst = pad_sequences(que_text_lst, maxlen=max_len_que, dtype=object, padding='post', truncating='post', value='')



if short_answer_end <= max_len_document:

    output_label[short_answer_start:short_answer_end] = np.ones(short_answer_end - short_answer_start)
from keras.preprocessing import text

def train_tokenizer(train_data):

    tokenizer = text.Tokenizer(num_words=50, filters='!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n', lower=True, char_level=True) #split='', 

    tokenizer.fit_on_texts(train_data)

    return tokenizer



tokenizer = train_tokenizer(doc_text_lst[0])

doc_text_lst[0] = tokenizer.texts_to_sequences(doc_text_lst[0])

doc_text_chars = pad_sequences(doc_text_lst[0], maxlen=max_len_word, padding='post', truncating='post', value=0)



que_text_lst[0] = tokenizer.texts_to_sequences(que_text_lst[0])

que_text_chars = pad_sequences(que_text_lst[0], maxlen=max_len_word, padding='post', truncating='post', value=0)
doc_text_chars




doc_text_chars.shape



que_text_chars




que_text_chars.shape



x_train = np.array(doc_text_chars)



lst = []



lst.append([[item] for item in doc_text_chars])

lst.append([[item] for item in doc_text_chars])

x_train = np.asarray(lst)



y_train = []

y_train.append([[item] for item in output_label])

y_train.append([[item] for item in output_label])

y_train = np.asarray(y_train)



# history = model.fit(x_train, y_train, epochs=5)
y_train.shape




x_train.shape



doc_input = keras.Input(shape=(max_len_document, 1, 40), name='doc_text')  #TODO -> make the length of the sequences variable



body_features = doc_input



body_features = layers.Reshape((max_len_document, 40))(body_features)



#Embed each character in the text into a 64-dimensional vector

body_features = layers.Embedding(50, 64)(body_features)



body_features = layers.TimeDistributed(layers.LSTM(64))(body_features)



short_answer = layers.TimeDistributed(layers.Dense(2, activation='softmax', name='short_answer'))(body_features)

# Instantiate an end-to-end model predicting both priority and department

model = keras.Model(inputs=doc_input, outputs=short_answer, name='qa_model')

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

model.summary()



history = model.fit(x_train, y_train, epochs=3)

doc_input = keras.Input(shape=(2500, 1, 40), name='doc_text')  #TODO -> make the length of the sequences variable



body_features = doc_input



body_features = layers.TimeDistributed(layers.LSTM(64))(body_features)



short_answer = layers.TimeDistributed(layers.Dense(2, activation='softmax', name='short_answer'))(body_features)

# Instantiate an end-to-end model predicting both priority and department

model = keras.Model(inputs=doc_input, outputs=short_answer, name='qa_model')

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

model.summary()



history = model.fit(x_train, y_train, epochs=10)



# model.summary()
#V2 ready to go

x_train = np.array(doc_text_chars)



lst = []

lst.append(doc_text_chars)

lst.append(doc_text_chars)

x_train = np.asarray(lst)





y_train = []

y_train.append(1950)

y_train.append(1950)

y_train = np.asarray(y_train)



# history = model.fit(x_train, y_train, epochs=5)



doc_input = keras.Input(shape=(2500, 40), name='doc_text')  #TODO -> make the length of the sequences variable



body_features = doc_input



#Embed each character in the text into a 64-dimensional vector

body_features = layers.Embedding(50, 64)(body_features)



# concolution 2d in order to process the input further

body_features = layers.Conv2D(32, 3, activation='relu')(body_features)



body_features = layers.Reshape((2498, 38 * 32))(body_features)



# Reduce sequence of embedded words in the body into a single 32-dimensional vector

body_features = layers.LSTM(64)(body_features)

# Stick a department classifier on top of the features

department_pred = layers.Dense(2500, activation='softmax', name='department')(body_features)

# Instantiate an end-to-end model predicting both priority and department

model = keras.Model(inputs=doc_input, outputs=department_pred, name='qa_model')

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])

model.summary()



history = model.fit(x_train, y_train, epochs=5)