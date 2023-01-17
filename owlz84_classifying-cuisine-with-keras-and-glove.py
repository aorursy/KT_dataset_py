# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import json

recipeRaw = pd.read_json("../input/whats-cooking/train.json")

recipeRaw["ingredientsFlat"] = recipeRaw["ingredients"].apply(lambda x: ' '.join(x))

recipeRaw.head()
recipeRawTest = pd.read_json("../input/whats-cooking/test.json")

recipeRawTest["ingredientsFlat"] = recipeRawTest["ingredients"].apply(lambda x: ' '.join(x))

testdocs = recipeRawTest["ingredientsFlat"].values

recipeRawCombined = recipeRaw.append(recipeRawTest)

recipeRawCombined[40000:].head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(recipeRaw["cuisine"].values)

list(le.classes_)
from keras.utils.np_utils import to_categorical

docs = recipeRaw["ingredientsFlat"].values

testdocs = recipeRawTest["ingredientsFlat"].values

docsCombined = recipeRawCombined["ingredientsFlat"].values

labels_enc = le.transform(recipeRaw["cuisine"].values)

labels = to_categorical(labels_enc)

labels
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



# prepare tokenizer

t = Tokenizer()

t.fit_on_texts(docsCombined)

vocab_size = len(t.word_index) + 1

# integer encode the documents

encoded_docs = t.texts_to_sequences(docs)

encoded_test_docs = t.texts_to_sequences(testdocs)

print(vocab_size)

# pad documents to a max length of 4 words

max_length = 40

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_length, padding='post')

print(len(padded_docs))
# load the whole embedding into memory

embeddings_index = dict()

f = open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
vocab = pd.DataFrame.from_dict(t.word_index,orient="index")

vocab.drop([0],axis=1).reset_index().rename(columns={"index":"word"}).to_csv("vocab.csv",index=False)
# create a weight matrix for words in training docs

embedding_matrix = np.zeros((vocab_size, 100))

for word, i in t.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
print(embedding_matrix.shape)
from keras.models import Sequential

from keras.layers import Dense, Flatten

from keras.layers.convolutional import Conv1D, MaxPooling1D

from keras.layers.embeddings import Embedding

from sklearn.model_selection import KFold



# fix random seed for reproducibility

seed = 42

np.random.seed(seed)



# define 10-fold cross validation test harness

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)



cvscores = []

for train, test in kfold.split(padded_docs, labels):

    # define the model



    model = Sequential()

    model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=40, trainable=False))

    model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(250, activation='relu'))

    model.add(Dense(le.classes_.size, activation='sigmoid'))

    # compile the model

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # summarize the model

    if cvscores == []:

        print(model.summary())

    # fit the model

    model.fit(padded_docs[train], labels[train], epochs=5, verbose=0)

    scores = model.evaluate(padded_docs[test], labels[test], verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
predictions = model.predict(padded_test_docs)

print(predictions.shape)

recipeRawTest["cuisine"] = [le.classes_[np.argmax(prediction)] for prediction in predictions]

recipeRawTest.head()

recipeRawTest.drop(["ingredients","ingredientsFlat"],axis=1).to_csv("submission.csv",index=False)