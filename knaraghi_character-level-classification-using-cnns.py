import numpy as np

import pandas as pd
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
x_train = train.iloc[:,1].values

y_train = train.iloc[:,2].values
import string



maxlen = 250

alphabet = (list(string.ascii_lowercase) + list(string.digits) +

                list(string.punctuation) + ['\n'])

vocab_size = len(alphabet)

check = set(alphabet)



vocab = {}

reverse_vocab = {}

for ix, t in enumerate(alphabet):

    vocab[t] = ix

    reverse_vocab[ix] = t



input_array = np.zeros((len(x_train), maxlen, vocab_size))

for i, sentence in enumerate(x_train):

    counter = 0

    sentence_array = np.zeros((maxlen, vocab_size))

    chars = list(sentence.lower().replace(' ', ''))

    for c in chars:

        if counter >= maxlen:

            pass

        else:

            char_array = np.zeros(vocab_size, dtype=np.int)

            if c in check:

                ix = vocab[c]

                char_array[ix] = 1

            sentence_array[counter, :] = char_array

            counter +=1

    input_array[i, :, :] = sentence_array
print(np.shape(input_array))
from sklearn.preprocessing import LabelBinarizer



one_hot = LabelBinarizer()

y_train = one_hot.fit_transform(y_train)

y_train
from keras.models import Sequential

from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Activation

from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D



model = Sequential()

model.add(Conv1D(filters=64, kernel_size=3, padding='same', input_shape=(250, 69)))

model.add(Conv1D(filters=64, kernel_size=3, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv1D(filters=64, kernel_size=3, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=3, strides=2))

model.add(Conv1D(filters=128, kernel_size=3, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv1D(filters=128, kernel_size=3, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=3, strides=2))

model.add(Conv1D(filters=256, kernel_size=3, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv1D(filters=256, kernel_size=3, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling1D(pool_size=3, strides=2))

model.add(Conv1D(filters=512, kernel_size=3, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Conv1D(filters=512, kernel_size=3, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(GlobalMaxPooling1D())

model.add(Dense(2048, activation='relu'))

model.add(Dense(2048, activation='relu'))

model.add(Dense(3, activation='softmax'))



model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(input_array, y_train, validation_split=0.2, epochs=40, batch_size=64, verbose=2)
x_test = test.iloc[:,1].values
test_array = np.zeros((len(x_test), maxlen, vocab_size))

for i, sentence in enumerate(x_test):

    counter = 0

    sentence_array = np.zeros((maxlen, vocab_size))

    chars = list(sentence.lower().replace(' ', ''))

    for c in chars:

        if counter >= maxlen:

            pass

        else:

            char_array = np.zeros(vocab_size, dtype=np.int)

            if c in check:

                ix = vocab[c]

                char_array[ix] = 1

            sentence_array[counter, :] = char_array

            counter +=1

    test_array[i, :, :] = sentence_array
print(np.shape(test_array))
y_test = model.predict_proba(test_array)
ids = test['id']
submission = pd.DataFrame(y_test, columns=['EAP', 'HPL', 'MWS'])

submission.insert(0, "id", ids)

submission.to_csv("submission.csv", index=False)