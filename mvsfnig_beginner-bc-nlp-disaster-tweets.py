import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)y

import os
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df_train.info()
df_train.head(10)
# count target 

sns.countplot(df_train['target'])
from keras.preprocessing.text import Tokenizer # create tokens

from keras import preprocessing  # convert list sequences to array numpy
# pre processing

from keras.utils.np_utils import to_categorical      # convert to one-hot-encoding

from sklearn.model_selection import train_test_split # spit data in train and test
NUM_WORDS = 1000

tokenizer = Tokenizer(num_words=NUM_WORDS)

tokenizer.fit_on_texts(df_train['text'])
sequences = tokenizer.texts_to_sequences(df_train['text'])

X = preprocessing.sequence.pad_sequences(sequences)

X = np.asarray(X).astype('float32')

MAX_LEN = X.shape[1]
def norm(data):

    return ((data - np.min(data)) / (np.max(data) - np.min(data)))
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
# define Y_TRAIN

Y = np.asarray(df_train['target'])

Y = to_categorical(Y, num_classes=2)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
from keras import models

from keras import layers
def SMNN(x_input):

    model = models.Sequential()

    model.add(layers.Dense(12, activation='tanh', input_shape=(x_input,)))

    model.add(layers.Dense(12, activation='tanh'))

    model.add(layers.Dense(2, activation='sigmoid'))

    

    # rmsprop, adam

    model.compile(optimizer='adam',

                  loss='binary_crossentropy',

                  metrics=['accuracy'])

    return model
model = SMNN(X_train.shape[1])
# train Neural Network

hist = model.fit(X_train,

                    Y_train,

                    epochs=30,

                    batch_size=512, 

                    validation_split=0.2, verbose=0)
df_fit = pd.DataFrame(hist.history)
fig = plt.figure(figsize=(20, 6))



ax = fig.add_subplot(121)

ax.plot(df_fit['loss'], '-o', label='loss')

ax.plot(df_fit['val_loss'], '--', label='validation')

ax.legend()



ax = fig.add_subplot(122)

ax.plot(df_fit['accuracy'], '-o', label='accuracy')

ax.plot(df_fit['val_accuracy'], '--', label='validation')

ax.legend()



plt.show()
scores = model.evaluate(X_test, Y_test)

print('loss .......: ', round(scores[0], 3))

print('acc ........: ', round(scores[1]*100, 2))
# predict 

y_pred  = model.predict(X_test)

y_pred.shape
# scores

from sklearn.metrics import f1_score
# computer metric F1

f1_score(np.argmax(Y_test, axis=1), np.argmax(y_pred, axis=1))
df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

df_test.head()
X_TEST = preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(df_test['text'].values), maxlen=MAX_LEN)

X_TEST = np.asarray(X_TEST).astype('float32')

X_TEST.shape
y_pred_test  = model.predict(X_TEST)
df_submission = pd.DataFrame({'id':df_test['id'].values, 'target':np.argmax(y_pred_test, axis=1)})

df_submission.head()
# save

df_submission.to_csv('submission.csv', encoding='utf-8', index=False)