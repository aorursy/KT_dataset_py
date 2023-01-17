import tensorflow as tf

from tensorflow import keras

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os

import nltk

from sklearn.utils import shuffle





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

train_df = shuffle(pd.read_csv("../input/nlp-getting-started/train.csv"))
train_df.shape
test_df.shape
test_df.head()
train_df.head()
train_df = shuffle(train_df)
word_index = {}

word_index["<PAD>"] = 0

word_index["<START>"] = 1

word_index["<UNK>"] = 2

index = 3
def add_to_index(txt):

    global index

    for word in nltk.word_tokenize(txt.lower()):

        if not word_index.get(word):

            word_index[word] = index

            index += 1

def encode(txt):

    ret = [0]

    for word in nltk.word_tokenize(txt.lower()):

        if not word_index.get(word):

            ret.append(word_index["<UNK>"])

        else:

            ret.append(word_index[word])

    return ret
for txt in train_df["text"]:

    add_to_index(txt)

train_df["encoded"] = [encode(txt) for txt in train_df["text"]]

test_df["encoded"] = [encode(txt) for txt in test_df["text"]]
max_len = max([len(x) for x in train_df["encoded"]])

vocab_size = len(word_index)

max_len, vocab_size
train_df["encoded"]
train_data = keras.preprocessing.sequence.pad_sequences(train_df["encoded"], value=word_index["<PAD>"], padding="post", maxlen=75)

test_data = keras.preprocessing.sequence.pad_sequences(test_df["encoded"], value=word_index["<PAD>"], padding="post", maxlen=75)
train_labels = train_df["target"]

#test_labels = test_df["target"]
train_data
model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, 10))

model.add(keras.layers.GlobalAveragePooling1D())

#model.add(keras.layers.Bidirectional(tf.keras.layers.LSTM(10)))

#model.add(keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)))

#model.add(keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))

model.add(keras.layers.Dense(10, activation="relu"))

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
x_val = train_data[:1013]

x_train = train_data[1013:]



y_val = train_labels[:1013]

y_train = train_labels[1013:]
N_VALIDATION = len(x_val)

N_TRAIN = len(x_train)

BATCH_SIZE = 256

STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

N_VALIDATION, N_TRAIN
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(

  0.001,

  decay_steps=STEPS_PER_EPOCH*1000,

  decay_rate=1,

  staircase=False)



def get_optimizer():

  return tf.keras.optimizers.Adam(lr_schedule)
#fitModel = model.fit(x_train, y_train, epochs=100, 

#                     batch_size=256, 

#                     validation_data=(x_val, y_val), verbose=1)

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):

  if optimizer is None:

    optimizer = get_optimizer()

  model.compile(optimizer=optimizer,

                loss='binary_crossentropy',

                metrics=['accuracy'])



  model.summary()

    

  history = model.fit(

    x_train,

    y_train,

    steps_per_epoch = STEPS_PER_EPOCH,

    epochs=max_epochs,

    validation_data=(x_val, y_val),

    verbose=1)

  return history
fitModel = compile_and_fit(model, 'test', max_epochs=60)
loss, accuracy = model.evaluate(x_val, y_val)



print("Loss: ", loss)

print("Accuracy: ", accuracy)
history_dict = fitModel.history

history_dict.keys()
import matplotlib.pyplot as plt



acc = history_dict['accuracy']

val_acc = history_dict['val_accuracy']

loss = history_dict['loss']

val_loss = history_dict['val_loss']



epochs = range(1, len(acc) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()   # clear figure



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(loc='lower right')



plt.show()
predict = model.predict(test_data)

predict[predict > 0.5] = 1

predict[predict <= 0.5] = 0

sample_submission["target"] = predict.astype("int")
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)