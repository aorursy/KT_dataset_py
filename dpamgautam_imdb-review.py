import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
import tensorflow as tf

print(tf.__version__)
tf.enable_eager_execution()
path = './../input/aclimdb/'
def shuffle(X, y):

    perm = np.random.permutation(len(X))

    X = X[perm]

    y = y[perm]

    return X, y
def load_imdb_dataset(path):

    imdb_path = os.path.join(path, 'aclImdb')



    # Load the dataset

    train_texts = []

    train_labels = []

    test_texts = []

    test_labels = []

    for dset in ['train', 'test']:

        for cat in ['pos', 'neg']:

            dset_path = os.path.join(imdb_path, dset, cat)

            for fname in sorted(os.listdir(dset_path)):

                if fname.endswith('.txt'):

                    with open(os.path.join(dset_path, fname)) as f:

                        if dset == 'train': train_texts.append(f.read())

                        else: test_texts.append(f.read())

                    label = 0 if cat == 'neg' else 1

                    if dset == 'train': train_labels.append(label)

                    else: test_labels.append(label)

                        

    # Converting to np.array

    train_texts = np.array(train_texts)

    train_labels = np.array(train_labels)

    test_texts = np.array(test_texts)

    test_labels = np.array(test_labels)



    # Shuffle the dataset

    train_texts, train_labels = shuffle(train_texts, train_labels)

    test_texts, test_labels = shuffle(test_texts, test_labels)



    # Return the dataset

    return train_texts, train_labels, test_texts, test_labels
training_sentences, training_labels, testing_sentences, testing_labels = load_imdb_dataset(path)
training_labels[:10]
testing_labels[:10]
import re



def preprocess(senti):

    l = []

    for each in senti:

        for k in each.split("\n"):

            s = re.sub(r"[^a-zA-Z0-9]+", ' ', k)

            s = re.sub("\d+", "", s)

            s = re.sub(' +', ' ', s)

            s = s.lower()

        l.append(s)

    return l



train_clean = preprocess(training_sentences)

test_clean = preprocess(testing_sentences)
from nltk.corpus import stopwords

en_stop_words = stopwords.words('english')
def remove_stop_words(corpus):

    removed_stop_words = []

    for senti in corpus:

        removed_stop_words.append( " ".join([word for word in senti.split() if not word in en_stop_words]) )

    return removed_stop_words



no_stop_train = remove_stop_words(train_clean)

no_stop_test = remove_stop_words(test_clean)
from nltk.stem import WordNetLemmatizer



def lemmatize(corpus):

    lemmatizer = WordNetLemmatizer()

    lemmatized = []

    for text in corpus:

        lemmatized.append( " ".join([lemmatizer.lemmatize(word) for word in text.split()]) )

    return lemmatized



lemmatized_train = lemmatize(no_stop_train)

lemmatized_test = lemmatize(no_stop_test)
vocab_size = 10000

embed_dim = 16

max_len = 120

trunc_type = 'post'

oov_tok = "oov"
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(lemmatized_train)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(lemmatized_train)

padded = pad_sequences(sequences, maxlen=max_len, truncating=trunc_type)



testing_sequences = tokenizer.texts_to_sequences(lemmatized_test)

testing_padded = pad_sequences(testing_sequences, maxlen=max_len)
import warnings

warnings.filterwarnings('ignore')



from sklearn.feature_extraction.text import CountVectorizer

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, Flatten

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
model = tf.keras.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_len))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(6, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
num_epoch = 2
padded.shape, training_labels.shape, testing_padded.shape, testing_labels.shape
model.fit(padded, training_labels, epochs=num_epoch, validation_data=(testing_padded, testing_labels))
e = model.layers[0]

weights = e.get_weights()[0]

print(weights.shape)
weights[:1]