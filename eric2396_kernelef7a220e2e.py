import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



df = pd.DataFrame()

df = pd.read_csv('../input/horror_train.csv', encoding='utf-8')

df.head(3)
import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



review_lines = list()

lines = df['text'].values.tolist()



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



EMBEDDING_DIM = 300

# train word2vec model

model = gensim.models.Word2Vec(sentences=review_lines, size=EMBEDDING_DIM, window=4, workers=4, min_count=1)

# vocab size

words = list(model.wv.vocab)

print('Vocabulary size: %d' % len(words))
model.wv.most_similar('gold')
# save model in ASCII (word2vec) format

filename = 'horror_embedding_word2vec.txt'

model.wv.save_word2vec_format(filename, binary=False)
import os



embeddings_index = {}

f = open(os.path.join('', 'horror_embedding_word2vec.txt'),  encoding = "utf-8")

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:])

    embeddings_index[word] = coefs

f.close()
total_reviews = df['text'].values

max_length = max([len(s.split()) for s in total_reviews])
print(max_length)
from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



VALIDATION_SPLIT = 0.2



# vectorize the text samples into a 2D integer tensor

tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(review_lines)

sequences = tokenizer_obj.texts_to_sequences(review_lines)



# pad sequences

word_index = tokenizer_obj.word_index

print('Found %s unique tokens.' % len(word_index))



text_pad = pad_sequences(sequences, maxlen=max_length)

labelencoder_X = LabelEncoder()

author = df['author'].values

author = pd.DataFrame(labelencoder_X.fit_transform(df['author']))

onehotencoder = OneHotEncoder(categorical_features = [0])

author = pd.DataFrame(onehotencoder.fit_transform(author).toarray())

author = np.array(author)

author = author.astype(dtype = 'int32')



print('Shape of review tensor:', text_pad.shape)

print('Shape of sentiment tensor:', author.shape)
# split the data into a training set and a validation set

indices = np.arange(text_pad.shape[0])

np.random.shuffle(indices)

text_pad = text_pad[indices]

author = author[indices]

num_validation_samples = int(VALIDATION_SPLIT * text_pad.shape[0])



X_train_pad = text_pad[:-num_validation_samples]

y_train = author[:-num_validation_samples]

X_test_pad = text_pad[-num_validation_samples:]

y_test = author[-num_validation_samples:]
print('Shape of X_train_pad tensor:', X_train_pad.shape)

print('Shape of y_train tensor:', y_train.shape)



print('Shape of X_test_pad tensor:', X_test_pad.shape)

print('Shape of y_test tensor:', y_test.shape)
EMBEDDING_DIM = 300

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

model.add(Dense(3, activation='softmax'))

print(model.summary())



# compile network

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# fit the model

model.fit(X_train_pad, y_train, batch_size=128, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)