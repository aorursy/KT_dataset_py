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
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np
df = pd.DataFrame()

df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv', encoding='utf-8')



df.head(3)
df.loc[df['sentiment'] == 'positive', 'sentiment'] = 1

df.loc[df['sentiment'] == 'negative', 'sentiment'] = 0

df.head()
import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



review_lines = list()

lines = df['review'].values.tolist()



for line in lines:   

    tokens = word_tokenize(line)

    # convert to lower case

    tokens = [w.lower() for w in tokens]

    # remove punctuation from each word    

    table = str.maketrans('', '', string.punctuation)

    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic

    words = [word for word in stripped if word.isalpha()]

    # filter out stop words    

    stop_words = set(stopwords.words('english'))

    words = [w for w in words if not w in stop_words]

    review_lines.append(words)
len(review_lines)
import gensim 



EMBEDDING_DIM = 100

# train word2vec model

model = gensim.models.Word2Vec(sentences=review_lines, size=EMBEDDING_DIM, window=5, workers=4, min_count=1)

# vocab size

words = list(model.wv.vocab)

print('Vocabulary size: %d' % len(words))
# save model in ASCII (word2vec) format

filename = 'imdb_embedding_word2vec.txt'

model.wv.save_word2vec_format(filename, binary=False)
# let us try some utility functions of gensim word2vec more details here 



model.wv.most_similar('horrible')#, topn =1)
#Let’s see the result of semantically reasonable word vectors (king - man + woman)

model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])


#Let’s see the result of semantically reasonable word vectors (king - man + woman)

model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
#odd word out

print(model.wv.doesnt_match("woman king queen movie".split()))
model.wv.similar_by_word("cat")
print(model.similarity('boy', 'girl'))
import os



embeddings_index = {}

f = open(os.path.join('', 'imdb_embedding_word2vec.txt'),  encoding = "utf-8")

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:])

    embeddings_index[word] = coefs

f.close()
X_train = df.loc[:24999, 'review'].values

y_train = df.loc[:24999, 'sentiment'].values

X_test = df.loc[25000:, 'review'].values

y_test = df.loc[25000:, 'sentiment'].values
total_reviews = X_train + X_test

max_length = 100 # try other options like mean of sentence lengths
from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences



VALIDATION_SPLIT = 0.2



# vectorize the text samples into a 2D integer tensor

tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(review_lines)

sequences = tokenizer_obj.texts_to_sequences(review_lines)



# pad sequences

word_index = tokenizer_obj.word_index

print('Found %s unique tokens.' % len(word_index))



review_pad = pad_sequences(sequences, maxlen=max_length)

sentiment =  df['sentiment'].values

print('Shape of review tensor:', review_pad.shape)

print('Shape of sentiment tensor:', sentiment.shape)



# split the data into a training set and a validation set

indices = np.arange(review_pad.shape[0])

np.random.shuffle(indices)

review_pad = review_pad[indices]

sentiment = sentiment[indices]

num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])



X_train_pad = review_pad[:-num_validation_samples]

y_train = sentiment[:-num_validation_samples]

X_test_pad = review_pad[-num_validation_samples:]

y_test = sentiment[-num_validation_samples:]
print('Shape of X_train_pad tensor:', X_train_pad.shape)

print('Shape of y_train tensor:', y_train.shape)



print('Shape of X_test_pad tensor:', X_test_pad.shape)

print('Shape of y_test tensor:', y_test.shape)
EMBEDDING_DIM =100

num_words = len(word_index) + 1

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))



for word, i in word_index.items():

    if i > num_words:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector
print(num_words)
from keras.models import Sequential

from keras.layers import Dense, Embedding, Flatten

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.initializers import Constant



# define model

model = Sequential()

# load pre-trained word embeddings into an Embedding layer

# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(num_words,

                            EMBEDDING_DIM,

                            embeddings_initializer=Constant(embedding_matrix),

                            input_length=max_length,

                            trainable=False)



model.add(embedding_layer)

model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

print(model.summary())



# compile network

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# fit the model

model.fit(X_train_pad, y_train, batch_size=128, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)


# evaluate the model

loss, accuracy = model.evaluate(X_test_pad, y_test, batch_size=128)

print('Accuracy: %f' % (accuracy*100))
#Let us test some  samples

# load the dataset but only keep the top n words, zero the rest



test_sample_1 = "This movie is fantastic! I really like it because it is so good!"

test_sample_2 = "Good movie!"

test_sample_3 = "Maybe I like this movie."

test_sample_4 = "Not to my taste, will skip and watch another movie"

test_sample_5 = "if you like action, then this movie might be good for you."

test_sample_6 = "Bad movie!"

test_sample_7 = "Not a good movie!"

test_sample_8 = "This movie really sucks! Can I get my money back please?"

test_samples = [test_sample_1, test_sample_2, test_sample_3, test_sample_4, test_sample_5, test_sample_6, test_sample_7, test_sample_8]



test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)

test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)



#predict

model.predict(x=test_samples_tokens_pad)
from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, GRU

from keras.layers.embeddings import Embedding

from keras.initializers import Constant



# define model

model = Sequential()

embedding_layer = Embedding(num_words,

                            EMBEDDING_DIM,

                            embeddings_initializer=Constant(embedding_matrix),

                            input_length=max_length,

                            trainable=False)

model.add(embedding_layer)

model.add(GRU(units=32,  dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))



# try using different optimizers and different optimizer configs

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



print('Summary of the built model...')

print(model.summary())
print('Train...')



model.fit(X_train_pad, y_train, batch_size=128, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)
print('Testing...')

score, acc = model.evaluate(X_test_pad, y_test, batch_size=128)



print('Test score:', score)

print('Test accuracy:', acc)



print("Accuracy: {0:.2%}".format(acc))