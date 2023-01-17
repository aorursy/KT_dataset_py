# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import layers

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/wine-dataset/train.csv')

test_data = pd.read_csv('/kaggle/input/wine-dataset/test.csv')

train_data.shape
train_data['country'] = train_data['country'].fillna('unknown')

train_data['region_1'] = train_data['region_1'].fillna('unknown')

train_data['province'] = train_data['province'].fillna('unknown')

train_data['price'] =  train_data['price'].fillna(train_data['price'].mean())

train_data['quality/price']  =np.array(np.log1p(train_data['points']))/np.array(np.log1p(train_data['price']))

train_data['value_for_money'] = train_data['quality/price'].apply(lambda val : 'High' if val > 1.5 else ('Medium' if val > 1.0 else 'Low'))
train_data['all_text_combined0'] = train_data['review_title'] +" " + train_data['review_description']

train_data['all_text_combined3'] = train_data['country'] +" " + train_data['province'] +" " + train_data['region_1'] +" " + train_data['review_title'] +" " + train_data['review_description']
train_data.head()
vocab_size =10000

embedding_dim =64

max_length =100

trunc_type = 'post'

padding_type ='post'

oov_token = '<OOV>'
X_train,X_test = train_test_split(train_data,test_size =0.09563618326,random_state =42) # to make no of smaples in test data divisible by 1024 in order to utizie TPU support
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(X_train['all_text_combined0'])

word_index = tokenizer.word_index
X_train_sequences = tokenizer.texts_to_sequences(X_train['all_text_combined0'])

print(X_train_sequences[1])

print(len(X_train_sequences))
X_train_padded = pad_sequences(X_train_sequences,truncating=trunc_type,padding=padding_type)

print(X_train_padded[1])

print(len(X_train_padded))
X_test_sequences = tokenizer.texts_to_sequences(X_test['all_text_combined0'])

print(X_test_sequences[1])
X_test_padded = pad_sequences(X_test_sequences,truncating=trunc_type,padding=padding_type)

print(X_test_padded[1])

print(len(X_test_padded))
lb = LabelEncoder()

Y_train_padded = np.array([[i] for i in lb.fit_transform(X_train['variety'])])

Y_test_padded = np.array([[i] for i in  lb.fit_transform(X_test['variety'])])

print(Y_train_padded.shape)

print(Y_test_padded.shape)
len(X_train_padded)//80
# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)





# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)



# instantiating the model in the strategy scope creates the model on the TPU

with tpu_strategy.scope():

    model = tf.keras.Sequential([

        tf.keras.layers.Embedding(vocab_size, embedding_dim),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),

        tf.keras.layers.Dense(embedding_dim, activation='relu'),

        tf.keras.layers.Dense(28, activation='softmax')

    ])

    

    model.summary()

    

    optim = tf.keras.optimizers.Adam(learning_rate=0.01)

    

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

num_epochs = 20

history = model.fit(X_train_padded, Y_train_padded, epochs=num_epochs, steps_per_epoch= 73,validation_data=(X_test_padded, Y_test_padded),validation_steps = 8, verbose=1)
def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
X_train,X_test = train_test_split(train_data,test_size =0.09563618326,random_state =42)
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(X_train['all_text_combined3'])

word_index = tokenizer.word_index
X_train_sequences = tokenizer.texts_to_sequences(X_train['all_text_combined3'])

X_train_padded = pad_sequences(X_train_sequences,truncating=trunc_type,padding=padding_type)

X_test_sequences = tokenizer.texts_to_sequences(X_test['all_text_combined3'])

X_test_padded = pad_sequences(X_test_sequences,truncating=trunc_type,padding=padding_type)
# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)





# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)



# instantiating the model in the strategy scope creates the model on the TPU

with tpu_strategy.scope():

    model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),

    tf.keras.layers.Dense(embedding_dim, activation='relu'),

    tf.keras.layers.Dense(28, activation='softmax')

])

    optim = tf.keras.optimizers.Adam(learning_rate=0.004)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
num_epochs = 20

history = model.fit(X_train_padded, Y_train_padded, epochs=num_epochs,steps_per_epoch=73, validation_data=(X_test_padded, Y_test_padded),validation_steps = 8, verbose=1)
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')