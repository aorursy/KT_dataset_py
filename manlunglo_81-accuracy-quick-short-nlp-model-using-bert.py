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

import seaborn as sns

from nltk.corpus import stopwords

import re

import string

import pandas_profiling

import random

# Install keras-bert packages

!pip install keras-bert

!pip install keras-rectified-adam
import codecs

import tensorflow as tf

from tqdm import tqdm

from chardet import detect

import keras

from keras_radam import RAdam

from keras import backend as K

from keras_bert import load_trained_model_from_checkpoint

import codecs

from keras_bert import Tokenizer
# Download bert model

!wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

!unzip -o uncased_L-12_H-768_A-12.zip
# Set parameters from bert

SEQ_LEN = 128

LR = 1e-4



# Set path of bert model

pretrained_path = 'uncased_L-12_H-768_A-12'

config_path = os.path.join(pretrained_path, 'bert_config.json')

checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')

vocab_path = os.path.join(pretrained_path, 'vocab.txt')

# Get bert model token

token_dict = {}

with codecs.open(vocab_path, 'r', 'utf8') as reader:

    for line in reader:

        token = line.strip()

        token_dict[token] = len(token_dict)

        

tokenizer = Tokenizer(token_dict)
data = pd.read_csv('../input/nlp-getting-started/train.csv')

data_eval = pd.read_csv('../input/nlp-getting-started/test.csv')



data.head()
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



def remove_tag(text):

    url = re.compile(r'(?<=\@)(.*?)(?= )')

    return url.sub(r'',text)



def remove_space(text):

    return text.replace(r'%20',' ')



def cleanse_data(data_in):

    data = data_in.copy()

    data['text'] = data['text'].apply(lambda x:remove_URL(x))

    data['text'] = data['text'].apply(lambda x:remove_html(x))

    data['text'] = data['text'].apply(lambda x:remove_emoji(x))

    data['text'] = data['text'].apply(lambda x:remove_tag(x))

    data['text'] = data['text'].apply(lambda x:remove_punct(x))

    

    data['keyword'].fillna('Nothing', inplace=True)

    data['keyword'] = data['keyword'].apply(lambda x:remove_space(x))

    data['keyword'] = data['keyword'].apply(lambda x:remove_punct(x))

    

    data['location'].fillna('Nothing', inplace=True)

    data['location'] = data['location'].apply(lambda x:remove_punct(x))

    return data

    
# Cleanse data

data_cleansed = cleanse_data(data)
# Get train dataset

def get_X(data, column_name = 'text'):

    X1 = [tokenizer.encode(text, max_len=SEQ_LEN)[0]  for text in data[column_name] ]

    X1 = np.array(X1)



    X2 = np.zeros_like(X1)

    X = [X1,X2]

    return X



X_train = get_X(data_cleansed)

y_train = np.array(data_cleansed.target)
# Load bert model

bert_model = load_trained_model_from_checkpoint(

      config_path,

      checkpoint_path,

      training=True,

      trainable=True,

      seq_len=SEQ_LEN,

  )
# Initialize keras session

sess = K.get_session()

uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])

init_op = tf.variables_initializer(

    [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]

)

sess.run(init_op)
# Build model structure

def get_model(neuron=256):

    

    inputs = bert_model.inputs[:2]

    outputs = bert_model.get_layer('NSP-Dense').output

    

    outputs = keras.layers.Dense(neuron, activation='relu')(outputs)

    outputs = keras.layers.Dense(units=1, activation='sigmoid')(outputs)



    model = keras.models.Model(inputs, outputs)

    # Make the last 4 layers trainable

    for layer in model.layers[:-20]:

        layer.trainable = False



    

    model.compile(RAdam(learning_rate =LR),loss='binary_crossentropy',metrics=['accuracy'])

    # model.summary()

    return model
# # Grid search the best configuration

# list_batch_size = [2,4,8,16,32,64,128,256]

# list_neuron = [8,16,32,64,128,256,512,1024,2048]



# result = []

# test_run = 10



# for i in range(test_run):

    

#     batch_size = random.choice(list_batch_size)

#     neuron = random.choice(list_neuron)

#     epochs = 8

    

#     print('-----------------------------------------------------------------')

#     print('[current config]:',[batch_size,neuron])

#     #val_accuracy = []

    

#     for j in range(2):

#         print('[current iteration]:',j+1)

#         model = get_model(neuron)

#         history = model.fit(

#                             X_train, 

#                             y_train, 

#                             batch_size=batch_size, 

#                             epochs=epochs, 

#                             verbose=1,

#                             validation_split=0.1

#                             )



#         for ep in range(epochs):

#             result.append([

#                             [batch_size,

#                             ep+1,

#                             neuron],

#                             history.history.get('val_acc')[ep],

#                             history.history.get('acc')[ep]

#                           ])
# X = []

# V = []

# A = []



# for x,val_accuracy,accuracy in result:

#     X.append(x)

#     V.append(val_accuracy)

#     A.append(accuracy)



# df_acc = pd.DataFrame(data={'combination':X, 'val_accuracy':V, 'accuracy':A})

# df_acc.to_csv('result.csv')
# Choose the best config for the model

model = get_model(neuron=32)

model.summary()



history = model.fit(X_train,y_train,

          epochs=4,

          batch_size=64, 

          verbose=1,

          validation_split=0.1)
# Compare training set pred with the actual result 

y_pred = model.predict(X_train, verbose=True).flatten()

y_pred = np.round(y_pred).astype(int).reshape(X_train[0].shape[0])



pred = pd.DataFrame(data=y_pred, columns=['pred'])

pred_df = pd.concat([data['text'], data_cleansed['text'], data['target'],pred], axis=1)

incorrect = pred_df[pred_df.pred!=pred_df.target]
incorrect
data_eval_cleansed = cleanse_data(data_eval)

X_eval = get_X(data_eval_cleansed)
y_eval = model.predict(X_eval, verbose=True)

y_eval = y_eval.flatten()

y_eval = np.round(y_eval).astype(int).reshape(X_eval[0].shape[0])
pred = pd.DataFrame(data=y_eval, columns=['pred'])

pd.concat([data_eval, data_eval_cleansed['text'], pred], axis=1)
output = pd.DataFrame({'id': data_eval_cleansed.id, 'target': y_eval})

output.to_csv('my_submission_20200205.csv', index=False)

print("Your submission was successfully saved!")