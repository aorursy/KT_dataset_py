import pandas as pd



data=pd.read_csv(r"../input/bbc-text.csv")
data.head()
from sklearn.preprocessing import LabelEncoder

df2 = pd.DataFrame()

df2["text"] = data["text"]

df2["label"] = LabelEncoder().fit_transform(data["category"])
import nltk

nltk.download('stopwords')



from nltk.corpus import stopwords

stop = stopwords.words('english')

df2['text'] = df2['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

df2['text'].head()
df2.head()
freq = pd.Series(' '.join(df2['text']).split()).value_counts()[-10:]

df2['text'] = df2['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

df2['text'].head()
import pandas as pd

import numpy as np

import spacy

from tqdm import tqdm

import re

import time

import pickle

pd.set_option('display.max_colwidth', 200)
import tensorflow_hub as hub

import tensorflow as tf



embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
import tensorflow as tf

import tensorflow_hub as hub

import pandas as pd

from sklearn import preprocessing

import keras

import numpy as np





y = list(df2['label'])

x = list(df2['text'])



le = preprocessing.LabelEncoder()

le.fit(y)



def encode(le, labels):

    enc = le.transform(labels)

    return keras.utils.to_categorical(enc)



def decode(le, one_hot):

    dec = np.argmax(one_hot, axis=1)

    return le.inverse_transform(dec)





x_enc = x

y_enc = encode(le, y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(np.asarray(x_enc), np.asarray(y_enc), test_size=0.2, random_state=42)
x_train.shape
from keras.layers import Input, Lambda, Dense

from keras.models import Model

import keras.backend as K



def ELMoEmbedding(x):

    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]



input_text = Input(shape=(1,), dtype=tf.string)

embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)

dense = Dense(256, activation='relu')(embedding)

pred = Dense(5, activation='softmax')(dense)

model = Model(inputs=[input_text], outputs=pred)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



with tf.Session() as session:

    K.set_session(session)

    session.run(tf.global_variables_initializer())  

    session.run(tf.tables_initializer())

    history = model.fit(x_train, y_train, epochs=1, batch_size=16)

    model.save_weights('./elmo-model.h5')



with tf.Session() as session:

    K.set_session(session)

    session.run(tf.global_variables_initializer())

    session.run(tf.tables_initializer())

    model.load_weights('./elmo-model.h5')  

    predicts = model.predict(x_test, batch_size=16)



y_test = decode(le, y_test)

y_preds = decode(le, predicts)



from sklearn import metrics



print(metrics.confusion_matrix(y_test, y_preds))



print(metrics.classification_report(y_test, y_preds))



from sklearn.metrics import accuracy_score



print("Accuracy of ELMO is:",accuracy_score(y_test,y_preds))