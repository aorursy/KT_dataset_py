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
import numpy as np

import tensorflow as tf

from tensorflow import keras

from keras.layers import Dense, Activation

from keras.layers.recurrent import SimpleRNN

from keras.models import Sequential

from keras.utils.vis_utils import plot_model
fin = open("../input/alice-in-wonderland-gutenbergproject/wonderland.txt", 'rb')

lines = []

for line in fin:

    line = line.strip().lower()

    line = line.decode("ascii", "ignore")

    if (line) == 0:

        continue

    lines.append(line)

fin.close()

# list --> string

text = " ".join(lines)
chars = set([c for c in text])

nb_chars = len(chars)

# chars map number index and number index map chars

char2index = dict((c, i) for i, c in enumerate(chars))

index2char = dict((i, c) for i, c in enumerate(chars))
# text data size each time

SEQLEN = 10

# steps each iteration

STEP = 1



input_chars = []

label_chars = []

for i in range(0, len(text) - SEQLEN, STEP):

    input_chars.append(text[i:i + SEQLEN]) # Not included text[i + SEQLEN]

    label_chars.append(text[i + SEQLEN]) # Include text[i + SEQLEN]
X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)

y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)

for i, input_char in enumerate(input_chars):

    for j, ch in enumerate(input_char):

        X[i, j, char2index[ch]] = 1

    y[i, char2index[label_chars[i]]] = 1
HIDDEN_SIZE = 128

BATCH_SIZE = 128

NUM_ITERATION = 25

NUM_EPOCHS_PER_ITERATION = 1

NUM_PREDS_PER_EPOCH = 100



model = Sequential()

model.add(

    SimpleRNN(HIDDEN_SIZE, return_sequences=False, input_shape=(SEQLEN, nb_chars), unroll=True)

)

model.add(Dense(nb_chars))

model.add(Activation("softmax"))



model.compile(

    loss="categorical_crossentropy",

    optimizer="rmsprop"

)
for iteration in range(NUM_ITERATION):

    print("=" * 50)

    print("Iteration #: %d" % (iteration))

    model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)

    

    test_idx = np.random.randint(len(input_chars))

    test_chars = input_chars[test_idx]

    print("Generating from seed: %s" % (test_chars))

    print(test_chars, end="")

    for i in range(NUM_PREDS_PER_EPOCH):

        Xtest = np.zeros((1, SEQLEN, nb_chars))

        for i, ch in enumerate(test_chars):

            Xtest[0, i, char2index[ch]] = 1

        pred = model.predict(Xtest, verbose=0)[0]

        ypred = index2char[np.argmax(pred)]

        print(ypred, end="")

        # 使用 test_chars + ypred继续

        test_chars = test_chars[1:] + ypred