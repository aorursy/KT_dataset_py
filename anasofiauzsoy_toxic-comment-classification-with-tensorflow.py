# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train= pd.read_csv("../input/jigsaw-toxic-comment-classification-cleaned-data/train_data.csv")

val = pd.read_csv("../input/jigsaw-toxic-comment-classification-cleaned-data/val_data.csv")

print(len(train), len(val))
train.columns.values
val.columns.values
translated_test = pd.read_csv("../input/jigsaw-toxic-comment-classification-cleaned-data/test_data.csv")

test = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test.csv")

print(len(translated_test), len(test))
translated_test.head()
train.head()
val.head()
dummy = train.cleaned_text.values[0]
translated_test[pd.isnull(translated_test.cleaned_text)]
translated_test.cleaned_text[pd.isnull(translated_test.cleaned_text)] = dummy

translated_test[pd.isnull(translated_test.cleaned_text)]
train = train[pd.notnull(train.cleaned_text)]
len(train[train.toxic == 0])
len(train[train.toxic == 1])
new_train = pd.concat((train[train.toxic == 1], train[train.toxic == 0].sample(100000)))
len(new_train)
import tensorflow as tf



vocab_size = 50000

max_length = 192
train_vals = train[['toxic']]

val_vals = val[['toxic']]

train_vals
len(val_vals)
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization



vectorize_layer = TextVectorization(

 max_tokens=vocab_size,

 output_mode='int',

 output_sequence_length=max_length)



vectorize_layer.adapt(np.concatenate((train.cleaned_text.values, val.cleaned_text.values, translated_test.cleaned_text.values)))
# from tensorflow.keras.models import Sequential

# from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, GRU, GlobalMaxPooling1D

# from tensorflow.keras.optimizers import Adam

# from tensorflow.keras import Input



# loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)



# def create_model():

#     model = Sequential()

#     model.add(tf.keras.Input(shape=(1,), dtype=tf.string))

#     model.add(vectorize_layer)

#     model.add(Embedding(vocab_size + 1, 64, input_length = max_length))

#     model.add(Bidirectional(LSTM(20, return_sequences = True)))

#     model.add(Bidirectional(LSTM(20, return_sequences = True)))

#     model.add(GlobalMaxPooling1D())

#     model.add(Dense(1, activation='sigmoid'))

#     model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.Adam(0.00005), metrics=['accuracy'])

#     return model

# #model.summary()

# model = create_model()
# history = model.fit(new_train.cleaned_text.values[::10],new_train.toxic.values[::10], epochs = 10, verbose = 1, 

#                     validation_data = (val.cleaned_text.values[::10], val_vals.values[::10]),

#                    callbacks = [tf.keras.callbacks.EarlyStopping(patience = 3)])
!pip install -U keras-tuner
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, GRU, GlobalMaxPooling1D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import Input

import kerastuner as kt



loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def create_model(hp):

    model = Sequential()

    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))

    model.add(vectorize_layer)

    model.add(Embedding(vocab_size + 1, hp.Int('units', min_value = 5, max_value = 200, step = 25), input_length = max_length))

#     model.add(tf.keras.layers.Conv1D(hp.Int('units', min_value = 5, max_value = 200, step = 25), 5, activation='relu'))

#     model.add(tf.keras.layers.GlobalMaxPooling1D())

    model.add(Bidirectional(LSTM(hp.Int('units', min_value = 5, max_value = 200, step = 25), return_sequences = True)))

    model.add(Bidirectional(LSTM(hp.Int('units', min_value = 5, max_value = 200, step = 25), return_sequences = True)))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values = [1e-4, 5e-5, 1e-5]) 

    model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.Adam(hp_learning_rate), metrics=['accuracy'])

    return model

#model.summary()



tuner = kt.Hyperband(create_model,

                     objective = 'val_accuracy', 

                     max_epochs = 15,

                     factor = 3)     



tuner.search(new_train.cleaned_text.values[::100],new_train.toxic.values[::100], epochs = 10,verbose = 2,

             validation_data = (val.cleaned_text.values[::100], val_vals.values[::100]), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

model = tuner.hypermodel.build(best_hps)



history = model.fit(new_train.cleaned_text.values,new_train.toxic.values, epochs = 20, verbose = 2, validation_data = (val.cleaned_text.values, val_vals.values),

                   callbacks = [tf.keras.callbacks.EarlyStopping(patience = 3)])
model.fit(val.cleaned_text.values, val_vals.values,epochs = 7, verbose = 2)
import matplotlib.pyplot as plt



def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.show()



plot_graphs(history, 'val_accuracy')

plot_graphs(history, 'loss')
translated_test.head()
test_toxic = model.predict(translated_test.cleaned_text.values)
evaluation = translated_test.id.copy().to_frame()

evaluation['toxic'] = np.round(test_toxic)

evaluation
evaluation.to_csv("submission.csv", index=False)