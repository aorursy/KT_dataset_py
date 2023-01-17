# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np 

import tensorflow as tf

import pandas as pd

import string 

import keras

from keras.preprocessing.text import Tokenizer

from keras import preprocessing

from keras.models import Sequential

from keras.layers import Flatten, Dense

from keras.layers import Embedding

from keras.utils.np_utils import to_categorical

from keras.utils import np_utils

from keras.datasets import imdb



from keras.preprocessing.text import one_hot

from keras.preprocessing.text import text_to_word_sequence

from keras.preprocessing import text, sequence



#from google.colab import drive



#import StringIO

import time

import sys

import csv
questions = pd.read_csv('../input/questions.csv')

answers = pd.read_csv('../input/answers.csv')
questions['questions_body'] = questions['questions_body'].str.replace("<p>", " ").str.replace("</p>", " ").str.replace("\n"," ")

answers['answers_body'] = answers['answers_body'].str.replace("<p>", " ").str.replace("</p>", " ").str.replace("\n"," ")

plt_data_questions = questions['questions_body']

plt_data_answers = answers['answers_body']

plt_data_questions_id = questions['questions_id']

plt_data_answers_questions_id = answers['answers_question_id']

plt_data_answers_author_id = answers['answers_author_id']
plt_data_questions.head(5)
df1 = pd.DataFrame([plt_data_questions_id,plt_data_questions]).T

df2 = pd.DataFrame([plt_data_answers_questions_id,plt_data_answers,plt_data_answers_author_id]).T
QA = pd.merge(df2, df1, left_on='answers_question_id',right_on='questions_id')
answers_body = QA['answers_body']

questions_body = QA['questions_body']

person_id = QA['answers_author_id']
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()

transfomed_label = encoder.fit_transform(person_id)
def preprocess(data):

    '''

    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution

    '''

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&' + '\n' 

    def clean_special_chars(text, punct):

        for p in punct:

            text = text.replace(p, ' ')

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))

    return data
a_train = preprocess(answers_body)

q_train = preprocess(questions_body)
tokenizer = Tokenizer()

tokenizer.fit_on_texts(list(a_train)+list(q_train))

a_train = tokenizer.texts_to_sequences(a_train)

q_train = tokenizer.texts_to_sequences(q_train)

question_len = 40 

a_train = sequence.pad_sequences(a_train,maxlen=400)

q_train = sequence.pad_sequences(q_train,maxlen=question_len)
import keras 

from keras import layers

from keras import backend as K 

from keras.models import Model

import numpy as np 

from keras import Input

from keras import Sequential

from keras import models
batch_size =16

latent_dim = 2 

question_shape = (40,)



input_qus = keras.Input(shape=question_shape)

#x = layers.LSTM(32,return_sequences=True,input_shape=(1,40))(input_qus)

x = layers.Dense(32,activation='relu',input_shape=(40,))(input_qus)

x = layers.Dense(16,activation='relu')(x)

#x = layers.LSTM((32))(x)

shape_before_flattening = K.int_shape(x)

z_mean = layers.Dense(10,activation='sigmoid',name='que_pool')(x)

#z_mean = layers.Dense(10,activation='relu',name='que_pool')(x)

z_log_var = layers.Dense(latent_dim)(x)



que_encoder = Model(input_qus,z_mean)
answer_shape = (400,)

input_ans = keras.Input(shape=answer_shape)

#x = layers.LSTM(64,return_sequences=True,input_shape=(1,400))(input_ans)

#x = layers.LSTM((32))(x)

x = layers.Dense(64,activation='relu',input_shape=(400,))(input_ans)

x = layers.Dense(32,activation='relu')(x)

x = layers.Dense(16,activation='relu')(x)

#x = layers.Dense(64,actiation='relu',input_shape=(400,1))

z_mean_ans = layers.Dense(10,activation='sigmoid',name='ans_pool')(x)

#z_mean_ans = layers.Dense(10,activation='relu',name='ans_pool')(x)

z_log_var_ans = layers.Dense(latent_dim)(x)



ans_encoder = Model(input_ans,z_mean_ans,name = 'ans_encoder')
d_input = Input(shape=(10,))

pid = layers.Dense(32,activation='relu')(d_input)

pid = layers.Dense(10169,activation='softmax')(pid)



decoder = Model(d_input,pid)



ans_encoder_output = ans_encoder.get_layer('ans_pool').output



person_id = decoder(ans_encoder_output)



full_model = Model(inputs=ans_encoder.input,outputs=person_id)
full_model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])
full_model.fit(a_train,transfomed_label,epochs=12,batch_size=128)
concatenated = layers.concatenate([z_mean,z_mean_ans],axis=-1)

combined = layers.Dense(1,input_shape=(1,))(concatenated)



correlation = Model([input_qus,input_ans],combined)
import keras.backend as K 



def custom_loss(z_mean,z_mean_ans):

  return K.mean(K.sum(K.square(z_mean-z_mean_ans)))
ans_encoder.trainable=False

correlation.compile(optimizer = 'rmsprop',loss=custom_loss,metrics=['acc'])
y_true = np.random.random((len(answers_body),1))

correlation.fit([q_train,a_train],y_true,epochs=10,batch_size=128)
que_encoder_output = que_encoder.get_layer('que_pool').output

que_person_id = decoder(que_encoder_output)



find_Mr_Right = Model(inputs=que_encoder.input,outputs=que_person_id)