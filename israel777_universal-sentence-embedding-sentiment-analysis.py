

from absl import logging



import tensorflow as tf

import tensorflow_hub as hub

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import re

import seaborn as sns



from textblob import TextBlob





import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras.optimizers import Adam,RMSprop,Adagrad

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Concatenate

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import layers
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
%%time

module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'

embed = hub.KerasLayer(module_url, trainable=False, name='USE_embedding')
# Import the Universal Sentence Encoder's TF Hub module

def get_sentiment(message):

    result = TextBlob(message).sentiment.subjectivity

    return result

def get_sentiment2(message):

    result = TextBlob(message).sentiment.polarity

    return result+1

# Reduce logging output.

def get_sentese_embedding(messages):

    return embed(messages)



def process_text(text):

    text = text.encode('ascii', errors='ignore').decode()

    text = text.lower()

    text = re.sub(r'http\S+', ' ', text)

    text = re.sub(r'#+', ' ', text )

    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)

    text = re.sub(r"([A-Za-z]+)'s", r"\1 is", text)

    text = re.sub(r"\'s", " ", text)     

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"won't", "will not ", text)

    text = re.sub(r"isn't", "is not ", text)

    text = re.sub(r"can't", "can not ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub('\W', ' ', text)

    text = re.sub(r'\d+', ' ', text)

    text = re.sub('\s+', ' ', text)

    text = text.strip()

    return text
train['text'] = train.text.apply(lambda x: process_text(x))

test['text'] = test.text.apply(lambda x: process_text(x))



train['count'] = train.text.apply(lambda x: len(x.split()))

test['count'] = test.text.apply(lambda x: len(x.split()))
sentiments_list = np.vectorize(get_sentiment)(train.text.values).reshape(-1,1)

sentiments_list2 = np.vectorize(get_sentiment2)(train.text.values).reshape(-1,1)

print(sentiments_list.shape)



test_sentiments_list = np.vectorize(get_sentiment)(test.text.values).reshape(-1,1)

test_sentiments_list2 = np.vectorize(get_sentiment2)(test.text.values).reshape(-1,1)

test_sentiments_list.shape
embeddings = get_sentese_embedding(train.text.values).numpy()

print(embeddings.shape)



test_embeddings = get_sentese_embedding(test.text.values).numpy()

print(test_embeddings.shape)
final = pd.concat([pd.DataFrame(embeddings),pd.DataFrame(sentiments_list),pd.DataFrame(sentiments_list2)],axis=1)

print(final.values.shape)



test_final = pd.concat([pd.DataFrame(test_embeddings),pd.DataFrame(test_sentiments_list),pd.DataFrame(test_sentiments_list2)],axis=1)

test_final.values.shape
train_data = final.values

train_labels = train.target.values





test_data = test_final.values

train_data.shape,train_labels.shape,test_data.shape
def build_model():

    model = Sequential([

        Input(shape=(514,)),

        Dense(256, activation='relu'),

        BatchNormalization(),

        Dropout(0.5),

        Dense(1, activation='sigmoid')

    ])

    model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model



model = build_model()

model.summary()



checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)



train_history = model.fit(

    train_data, train_labels,

    validation_split=0.3,

    epochs=100,

    callbacks=[checkpoint,callback],

    batch_size=128

)
model.load_weights('model.h5')

test_pred = model.predict(test_data)
submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)