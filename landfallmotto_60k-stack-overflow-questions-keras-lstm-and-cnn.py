# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf



from tensorflow.keras import layers

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing import text

from sklearn.metrics import classification_report



from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.feature_extraction.text import CountVectorizer

from yellowbrick.style import set_palette

from yellowbrick.text import FreqDistVisualizer
tf.test.gpu_device_name()
data=pd.read_csv('../input/60k-stack-overflow-questions-with-quality-rate/data.csv')
data.head(5)
data.Y.value_counts().plot.bar()
data['Text']=data.Body.apply(lambda x: BeautifulSoup(x, 'html.parser').text)

data.head(5)
data['Text']=data['Text'].str.lower()

data.head(5)
MAX_FEATURES = 20000

MAX_LEN = 200
X=data['Text'].values



plt.style.use('seaborn')



tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_FEATURES)

tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)



word_index = tokenizer.word_index



result = [len(x.split()) for x in X]





plt.figure(figsize=(20,5))

plt.title('Document size')

plt.hist(result, 200, density=False, range=(0,np.max(result)))

plt.show()





print('max length: %i / min length: %i / mean length: %i / limit length: %i' % (np.max(result), np.min(result), np.mean(result), MAX_LEN))

vectorizer = CountVectorizer()

docs       = vectorizer.fit_transform(X)

features   = vectorizer.get_feature_names()



visualizer = FreqDistVisualizer(features=features, orient='v',color='rb')

visualizer.fit(docs)

visualizer.show()
from sklearn.model_selection import train_test_split

train,validation=train_test_split(data,test_size=0.25, random_state=55)
EPOCHS = 25

BATCH_SIZE = 24

#MAX_LEN = 192
encoder = LabelEncoder()

encoder.fit(data.Y.values)

encoded_Y_train = encoder.transform(train.Y.values)

encoded_Y_valid = encoder.transform(validation.Y.values)





x_train = train.Text.values

x_valid = validation.Text.values





y_train = np_utils.to_categorical(encoded_Y_train)

y_valid = np_utils.to_categorical(encoded_Y_valid)
tokens=text.Tokenizer(num_words=MAX_FEATURES, lower=True)

tokens.fit_on_texts(list(x_train))
x_train=tokens.texts_to_sequences(x_train)

x_valid=tokens.texts_to_sequences(x_valid)





x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_valid = tf.keras.preprocessing.sequence.pad_sequences(x_valid, maxlen=MAX_LEN)
inputs = tf.keras.Input(shape=(None,), dtype="int32")

x = layers.Embedding(MAX_FEATURES, 128)(inputs)

x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

x = layers.Bidirectional(layers.LSTM(64))(x)

#x = layers.Flatten()(x)

#x = layers.Dropout(0.5)(x)

outputs = layers.Dense(3, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.summary()
inputs_cnn = tf.keras.Input(shape=(None,), dtype="int32")

x_cnn = layers.Embedding(MAX_FEATURES, 128)(inputs_cnn)

x_cnn = layers.Bidirectional(layers.GRU(128, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x_cnn)

x_cnn = layers.Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x_cnn)

avg_pool = layers.GlobalAveragePooling1D()(x_cnn)

max_pool = layers.GlobalMaxPooling1D()(x_cnn)

x_cnn = layers.concatenate([avg_pool, max_pool])



outputs_cnn = layers.Dense(3, activation="softmax")(x_cnn)

model_cnn = tf.keras.Model(inputs_cnn, outputs_cnn)

model_cnn.summary()
SGD=tf.keras.optimizers.SGD(learning_rate=0.01)

model.compile(loss='categorical_crossentropy',optimizer=SGD,metrics=[tf.keras.metrics.AUC()])
es_cb = EarlyStopping(monitor='val_loss', min_delta=0,  patience=10, verbose=0, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_valid, y_valid),callbacks = [es_cb,reduce_lr], verbose=1)
plt.plot(history.history['auc'])

plt.plot(history.history['val_auc'])
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])
test_question = ['I have sql server management studio 14.0.17825.0 and would like to use group_concat function. But I get error when i try to use. The error is invalid column name group_concat Is there any other function that I could use? Could you provide a sample code which could achieve what function group_concat?']
seq = tokenizer.texts_to_sequences(test_question)

padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)

pred = model.predict(padded)
labels=list(encoder.classes_)

print(np.argmax(pred), labels[np.argmax(pred)])
y_pred=model.predict(x_valid)

y_pred=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_valid,axis=1)
target_names = list(encoder.classes_)



print(classification_report(y_true, y_pred, target_names=target_names))