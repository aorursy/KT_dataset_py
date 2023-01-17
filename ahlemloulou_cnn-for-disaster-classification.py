# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from numpy import array

from numpy import asarray

from numpy import zeros



import re

import nltk

from nltk.corpus import stopwords



from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten

from keras.layers import GlobalMaxPooling1D

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
dataset = pd.read_csv("/kaggle/input/dataset/train.csv")

dataset.head()
datatest=pd.read_csv('/kaggle/input/dataset/test.csv', delimiter=',')

datatest.head()
import seaborn as sns



sns.countplot(x='target', data= dataset)
def preprocess_text(sen):

    # Removing html tags

    text = remove_tags(sen)



    # Remove punctuations and numbers

    text = re.sub('[^a-zA-Z]', ' ', text)



    # Single character removal

    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)



    # Removing multiple spaces

    text = re.sub(r'\s+', ' ', text)



    return text
TAG_RE = re.compile(r'<[^>]+>')



def remove_tags(text):

    return TAG_RE.sub('', text)
X = []

text = list(dataset['text'])

for sen in text:

    X.append(preprocess_text(sen))
y = dataset['target']

y = np.array(list(map(lambda x: 1 if x=="disaster" else 0, y)))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(X_train)



X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)
# Adding 1 because of reserved 0 index

vocab_size = len(tokenizer.word_index) + 1



maxlen = 100



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
embeddings_dictionary = dict()

glove_file = open('/kaggle/input/glove100/glove.6B.100d.txt', encoding="utf8")



for line in glove_file:

    records = line.split()

    word = records[0]

    vector_dimensions = asarray(records[1:], dtype='float32')

    embeddings_dictionary [word] = vector_dimensions

glove_file.close()
embedding_matrix = zeros((vocab_size, 100))

for word, index in tokenizer.word_index.items():

    embedding_vector = embeddings_dictionary.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector
from keras.layers.convolutional import Conv1D



model = Sequential()



embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)

model.add(embedding_layer)

model.add(Conv1D(128, 5, activation='relu'))

model.add(GlobalMaxPooling1D())

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train,

                    epochs=4,

                    verbose=True,

                    validation_data=(X_test,y_test),

                    batch_size=64)

loss, accuracy = model.evaluate(X_train, y_train, verbose=True)

print("Training Accuracy: {:.4f}".format(accuracy))

loss_val, accuracy_val = model.evaluate(X_test, y_test, verbose=True)

print("Testing Accuracy:  {:.4f}".format(accuracy_val))
test = []

text = list(datatest['text'])

for sen in text:

    test.append(preprocess_text(sen))
tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(test)

test = tokenizer.texts_to_sequences(test)
# Adding 1 because of reserved 0 index

vocab_size = len(tokenizer.word_index) + 1



maxlen = 100



test = pad_sequences(test, padding='post', maxlen=maxlen)

predictions = model.predict(test)
submission = pd.read_csv('/kaggle/input/dataset/sample_submission.csv')

submission["target"] = predictions

submission["target"] = submission["target"].apply(lambda predictions : 0 if predictions <=0.5 else 1)

submission.to_csv("submission.csv", index=False)

submission