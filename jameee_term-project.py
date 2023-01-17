import pandas as pd

data_frame = pd.read_csv('./boardgamegeek-reviews/bgg-13m-reviews.csv', index_col=0)
data_frame.drop(data_frame.columns[4], axis=1, inplace=True)
data_frame.drop(data_frame.columns[3], axis=1, inplace=True)
data_frame.drop(data_frame.columns[0], axis=1, inplace=True)
data_frame.head()
data_frame = data_frame[~data_frame.comment.str.contains("NaN",na=True)]
print(data_frame.head())
print('data shape: ', data_frame.shape)
data_frame["rating"] = data_frame["rating"].round(0).astype(int)
data_frame.groupby(["rating"]).count()
import matplotlib.pyplot as plt
plt.hist(data_frame.rating, 50)
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import string
from tqdm import tqdm
import numpy as np
import random
nltk.download('punkt')
nltk.download('stopwords')

x = np.array(data_frame.comment)
y = np.array(data_frame.rating)
y = np.round(y)

all_texts = []
for index, text in tqdm(enumerate(x)):
    # lower case
    text = text.lower()
    # tokenize
    words = word_tokenize(text)
    # topwords
    words = [w for w in words if w not in stopwords.words('english')]
    # remove punctuation
    words = [w for w in words if w not in string.punctuation]
    # Stemming
    words = [PorterStemmer().stem(w) for w in words]
    all_texts.append(words)

x = np.array(all_texts)
index = [i for i in range(len(x))]
random.shuffle(index)
x = x[index]
y = y[index]

np.save('x.npy', x)
np.save('y.npy', y)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import sklearn.metrics as metrics
import numpy as np
from tqdm import tqdm

print('load data')

x = np.load('x.npy', allow_pickle=True)
y = np.load('y.npy', allow_pickle=True)

for i, d in tqdm(enumerate(x)):
    sentence = ' '.join(x[i])
    x[i] = sentence
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


count_vect = CountVectorizer(min_df=0.001, max_df=0.5, max_features=1000)
X_train_counts = count_vect.fit_transform(x_train)
X_test_counts = count_vect.fit_transform(x_test)

count_vect = TfidfTransformer()
tf_transformer = TfidfTransformer().fit(X_train_counts)
x_train = tf_transformer.transform(X_train_counts)
x_train = x_train.toarray()
print(x_train.shape)

tf_transformer = TfidfTransformer().fit(X_test_counts)
x_test = tf_transformer.transform(X_test_counts)

def decision_tree(train_x, train_y, test_x, test_y):
    print('...Decision_tree...')
    clf = DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_split=20, min_samples_leaf=5).fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    print('Decision tree accuracy: ', metrics.accuracy_score(test_y, predict_y))


decision_tree(x_train, y_train, x_test, y_test)
from sklearn.ensemble import AdaBoostClassifier

def adaboost(train_x, train_y, test_x, test_y):
    print('...adaboost...')
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME.R",n_estimators=50, learning_rate=0.8).fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    print('Adaboost accuracy: ', metrics.accuracy_score(test_y, predict_y))
    
adaboost(x_train, y_train, x_test, y_test)
def bayes_fit_and_predicted(train_x, train_y, test_x, test_y):
    print('...Bayes...')
    clf = MultinomialNB().fit(train_x, train_y, alpha=1.0)
    predict_y = clf.predict(test_x)
    print('Bayes accuracy: ', metrics.accuracy_score(test_y, predict_y))
    
bayes_fit_and_predicted(x_train, y_train, x_test, y_test)
def svm_fit_and_predicted(train_x, train_y, test_x, test_y, C=1.0):
    print('...SVM...')
    clf = LinearSVC(C=1.0, penalty='l2').fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    print('SVM accuracy: ', metrics.accuracy_score(test_y, predict_y))

svm_fit_and_predicted(x_train, y_train, x_test, y_test)  
import numpy as np
import tensorflow as tf
import os
import sklearn.metrics as metrics
import datetime
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

print('load data')

x = np.load('x.npy', allow_pickle=True)
y = np.load('y.npy', allow_pickle=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)

# {word1: index1, word2: index2}
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)


num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM,
                                    input_length=MAX_SEQUENCE_LENGTH))
model.add(tf.keras.layers.GRU(256, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GRU(128))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(11, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# model.fit(x_train, y_train, batch_size=128, epochs=10,validation_split=VALIDATION_SPLIT)
model.fit(x_train, y_train, batch_size=128, epochs=5)
x_test = tokenizer.texts_to_sequences(x_test)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
# model save
model.save('gru_model.h5')

results = model.evaluate(x_test, y_test)
print('Test loss:', results[0])
print('Test accuracy:', results[1])
import numpy as np
import gensim
import logging
from gensim.models import Word2Vec
from nltk import sent_tokenize, word_tokenize, pos_tag
import numpy as np
import tensorflow as tf
import os
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import multi_gpu_model
import pickle

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


print('load data')

x = np.load('x.npy', allow_pickle=True)
y = np.load('y.npy', allow_pickle=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# model = Word2Vec(x_train, size=100, min_count=1, window=5)
# model.save('Word2Vec2.dict')
word2vec_model = Word2Vec.load('Word2Vec2.dict')

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)

# {word1: index1, word2: index2}
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)


print('Preparing embedding matrix.')
# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = word2vec_model[word]
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM,
                                    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                    mask_zero=True,
                                    input_length=MAX_SEQUENCE_LENGTH))
model.add(tf.keras.layers.GRU(256, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GRU(128))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(11, activation='softmax'))

# print(model.layers[0])
# model.layers[0].trainable = False
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
# model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.2)
model.fit(x_train, y_train, batch_size=256, epochs=3)

# model save
model.save('gru_word2vec.h5')

x_test = tokenizer.texts_to_sequences(x_test)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

results = model.evaluate(x_test, y_test)
print('Test loss:', results[0])
print('Test accuracy:', results[1])

import numpy as np
import tensorflow as tf
import os
import sklearn.metrics as metrics
import datetime
from sklearn.model_selection import train_test_split
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

print('load data')

x = np.load('x.npy', allow_pickle=True)
y = np.load('y.npy', allow_pickle=True)


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train, y_train = x, y

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_train)

fn = 'tokenizer.pkl'
with open(fn, 'wb') as f:
    picklestring = pickle.dump(tokenizer, f)


x_train = tokenizer.texts_to_sequences(x_train)

# {word1: index1, word2: index2}
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)


num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM,
                                    input_length=MAX_SEQUENCE_LENGTH))
model.add(tf.keras.layers.GRU(256, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GRU(128))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(11, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# model.fit(x_train, y_train, batch_size=128, epochs=10,validation_split=VALIDATION_SPLIT)
model.fit(x_train, y_train, batch_size=128, epochs=5)
# model save
model.save('gru_model.h5')

import random
import numpy as np
from tqdm import tqdm
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

data_frame = pd.read_csv('./boardgamegeek-reviews/bgg-13m-reviews.csv', index_col=0)
data_frame.drop(data_frame.columns[4], axis=1, inplace=True)
data_frame.drop(data_frame.columns[3], axis=1, inplace=True)
data_frame.drop(data_frame.columns[0], axis=1, inplace=True)


data_frame = data_frame[~data_frame.comment.str.contains("NaN", na=True)]
print(data_frame.head())
print('data shape: ', data_frame.shape)

data_frame["rating"] = data_frame["rating"].round(0).astype(int)
# print(data_frame.groupby(["rating"]).count())

rating_subset = data_frame[data_frame['rating'] == 1]
balance_df = rating_subset.sample(20000)

for i in range(9):
    rating_subset = data_frame[data_frame['rating'] == (i+2)]
    r = rating_subset.sample(20000)
    balance_df = balance_df.append(r)

print(balance_df.groupby(["rating"]).count())

nltk.download('punkt')
nltk.download('stopwords')

x = np.array(balance_df.comment)
y = np.array(balance_df.rating)

all_texts = []
for index, text in tqdm(enumerate(x)):
    # lower case
    text = text.lower()
    # tokenize
    words = word_tokenize(text)
    # topwords
    words = [w for w in words if w not in stopwords.words('english')]
    # remove punctuation
    words = [w for w in words if w not in string.punctuation]
    # Stemming
    words = [PorterStemmer().stem(w) for w in words]
    all_texts.append(words)

x = np.array(all_texts)
index = [i for i in range(len(x))]
random.shuffle(index)
x = x[index]
y = y[index]

np.save('balance_x.npy', x)
np.save('balance_y.npy', y)


import numpy as np
import tensorflow as tf
import os
import sklearn.metrics as metrics
import datetime
from sklearn.model_selection import train_test_split
import pickle


os.environ['CUDA_VISIBLE_DEVICES'] = "3"

MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 1000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

print('load data')

x = np.load('balance_x.npy', allow_pickle=True)
y = np.load('balance_y.npy', allow_pickle=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# x_train, y_train = x, y

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_train)

fn = 'balance_tokenizer.pkl'
with open(fn, 'wb') as f:
    picklestring = pickle.dump(tokenizer, f)


x_train = tokenizer.texts_to_sequences(x_train)

# {word1: index1, word2: index2}
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=EMBEDDING_DIM,
                                    input_length=MAX_SEQUENCE_LENGTH))
model.add(tf.keras.layers.GRU(256, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GRU(128))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(11, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=128, epochs=5,validation_split=VALIDATION_SPLIT)
# model.fit(x_train, y_train, batch_size=128, epochs=5)
# model save
# model.save('balance_gru_model.h5')

x_test = tokenizer.texts_to_sequences(x_test)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
results = model.evaluate(x_test, y_test)
print('Test loss:', results[0])
print('Test accuracy:', results[1])
