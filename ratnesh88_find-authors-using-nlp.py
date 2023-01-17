import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_hub as hub

import nltk
from nltk.probability import FreqDist 
import os
print(os.listdir("../input"))

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Any results you write to the current directory are saved as output.
# embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
embed = hub.Module("https://tfhub.dev/google/elmo/1")
train = pd.read_csv('../input/train.csv')
train.head()
test = pd.read_csv('../input/test.csv')
test.head()
sns.countplot(train['author']);
x = list(train['text'])
y = list(train['author'])
le = LabelEncoder()
le.fit(y)
le.classes_
def encode(le, label):
    enc = le.transform(label)
    return keras.utils.to_categorical(enc)

def decode(le, one_hot):
    dec = np.argmax(one_hot, axis=1)
    return le.inverse_transform(dec)
encD =  encode(le, ['EAP'])
encD
decD = decode(le, encD)
decD
x_enc = x
y_enc = encode(le , y)
y_enc
len(x)
x_train = np.asarray(x_enc[:16000])
y_train = np.asarray(y_enc[:16000])

x_test = np.asarray(x_enc[16000:])
y_test = np.asarray(y_enc[16000:])
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Lambda, Input
import keras.backend as K
def UniversalEmbedding(x):
#     return embed(tf.squeeze(tf.cast(x, tf.string)))
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(UniversalEmbedding, output_shape=(1024,))(input_text)
dense1 = Dense(256, activation='relu')(embedding)
dense2 = Dense(128, activation='relu')(dense1)
dropout = Dropout(0.5)
dense3 = Dense(128, activation='relu')(dense2)
pred = Dense(3, activation='softmax')(dense3)
model = Model([input_text], outputs=pred )

model.summary()
model.compile(loss= 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model_history = model.fit(x_train, y_train, batch_size=16, epochs=4)
    model.save_weights('./author_model_elmo.h5')
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    model.load_weights('./author_model_elmo.h5')
    score = model.evaluate(x_test, y_test, batch_size=16)
    test_pred = model.predict(test['text'], batch_size=16)
    pred = model.predict(x_test, batch_size=16)
score
preds = decode(le, pred)
y_true = decode(le, y_test)
preds
c= confusion_matrix(y_true=y_true, y_pred=preds)
sns.heatmap(c, annot=True, xticklabels=le.classes_, yticklabels=le.classes_);
print(classification_report(y_true=y_true, y_pred=preds))
submission = pd.read_csv('../input/sample_submission.csv')
submission.columns
test.head()
k = pd.DataFrame(test_pred, columns=['EAP', 'HPL', 'MWS'])
submission['EAP'] = k['EAP']
submission['HPL'] = k['HPL']
submission['MWS'] = k['MWS']
submission.head()
submission.to_csv('my_submission4.csv', index=False)