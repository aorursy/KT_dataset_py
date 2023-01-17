# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

# Any results you write to the current directory are saved as output.
from wordcloud import WordCloud

from sklearn.preprocessing import LabelEncoder

from keras.models import Model

from keras.layers import LSTM, Dense, Input, Dropout, Bidirectional, GlobalMaxPool1D, Embedding

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical
data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
data.head()
data.author.value_counts().plot(kind = 'bar')
data_length = data.text.apply(len)

data_length.head()
plt.figure(figsize = (12, 5))

plt.hist(data_length, bins = 20, range = [0, 500], color = 'r', alpha = 0.3)

plt.show()
data_split_length = data.text.apply(lambda x : len(x.split(" ")))

data_split_length.head()
plt.figure(figsize = (12, 5))

plt.hist(data_split_length, bins = 10, range = [0, 100], color = 'g', alpha = 0.5)

plt.show()
print("data_length max : ", np.max(data_length))

print("data_length min : ", np.min(data_length))

print("data_length mean : ", np.mean(data_length))

print("data_length 75% : ", np.percentile(data_length, 75))

print("data_length 90% : ", np.percentile(data_length, 90))
print("data_split_length max : ", np.max(data_split_length))

print("data_split_length min : ", np.min(data_split_length))

print("data_split_length mean : ", np.mean(data_split_length))

print("data_split_length 75% : ", np.percentile(data_split_length, 75))

print("data_split_length 90% : ", np.percentile(data_split_length, 90))
cloud = WordCloud(width = 400, height = 200).generate(" ".join(data.text))

plt.figure(figsize = (12, 5))

plt.imshow(cloud)

plt.axis('off')
cloud = WordCloud(width = 400, height = 200).generate(" ".join(data[data["author"] == 'HPL']['text']))

plt.figure(figsize = (12, 5))

plt.imshow(cloud)

plt.axis('off')
cloud = WordCloud(width = 400, height = 200).generate(" ".join(data[data["author"] == 'MWS']['text']))

plt.figure(figsize = (12, 5))

plt.imshow(cloud)

plt.axis('off')
cloud = WordCloud(width = 400, height = 200).generate(" ".join(data[data["author"] == 'EAP']['text']))

plt.figure(figsize = (12, 5))

plt.imshow(cloud)

plt.axis('off')
le = LabelEncoder()

le.fit(data.author)

y = le.transform(data.author)
y[:10]
y = to_categorical(y)
y[:10]
num_words = 5000

max_len = 50

emb_size = 64
tok = Tokenizer(num_words = num_words)

tok.fit_on_texts(list(data.text))
X = tok.texts_to_sequences(data.text)

X_test = tok.texts_to_sequences(test_data.text)
X = sequence.pad_sequences(X, maxlen = max_len)

X_test = sequence.pad_sequences(X_test, maxlen = max_len)
X[0]
def model():

    inp = Input(shape = (max_len, ))

    layer = Embedding(num_words, emb_size)(inp)

    layer = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout = 0.2))(layer)

    layer = GlobalMaxPool1D()(layer)

    layer = Dropout(0.2)(layer)

    layer = Dense(16, activation = 'relu')(layer)

    layer = Dropout(0.2)(layer)

    layer = Dense(3, activation = 'softmax')(layer)

    model = Model(inputs = inp, outputs = layer)

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model
model = model()

model.summary()
early_stop = EarlyStopping(monitor = 'val_loss', patience = 1)
hist = model.fit(X, y, batch_size = 32, epochs = 2, validation_split = 0.2, callbacks = [early_stop])
vloss = hist.history['val_loss']

loss = hist.history['loss']



x_len = np.arange(len(loss))



plt.plot(x_len, vloss, marker = '.', color = 'r', label = 'val_loss')

plt.plot(x_len, loss, marker = '.', color = 'b', label = 'loss')

plt.legend()

plt.grid()

plt.xlabel("epochs")

plt.ylabel("loss")

plt.show()
results = model.predict(X_test)

ids = test_data['id']

results = pd.DataFrame(results, columns=['EAP', 'HPL','MWS'])

results.insert(0, "id", ids)

results.head()
results.to_csv("my_submission.csv", index=False)