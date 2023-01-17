# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import gensim



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import roc_auc_score



from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from tensorflow.python.keras.callbacks import EarlyStopping



from keras.optimizers import Adam



import keras.models

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, GRU, BatchNormalization, GlobalMaxPooling1D, Dropout

from keras.layers.embeddings import Embedding

from keras.initializers import Constant



import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
!ls ../input
df = pd.read_csv('../input/imdb-reviews-dataset/imdb_reviews_dataset.csv')
model_weights = '../input/model-imdb/model.h5'
df['type_review'] = df['id'].apply(lambda x: x.split('_')[0])
df['type_review'].unique()
df_labeled = df[df['type_review'] != 'unsup'].copy() 
df_labeled['sentiment'] = df_labeled['type_review'].apply(lambda x: 1 if x == 'pos' else 0)
df_labeled['len_text'] = df_labeled['text'].apply(lambda x: len(x.split()))

df_labeled
len_text_info = df_labeled['len_text'].describe()

len_text_info
# set max len for padding

max_length = int(len_text_info['mean'] + 2 * len_text_info['std'])

print(max_length) # = 200
VALIDATION_SPLIT = 0.2



indices = np.arange(df_labeled.shape[0])

np.random.shuffle(indices)

review = df_labeled.iloc[indices]

num_validation_samples = int(VALIDATION_SPLIT * review.shape[0])



X_train = review.iloc[:-num_validation_samples]['text'].values

y_train = review.iloc[:-num_validation_samples]['sentiment'].values

X_test = review.iloc[-num_validation_samples:]['text'].values

y_test = review.iloc[-num_validation_samples:]['sentiment'].values
print(len(X_train), len(y_train))

print(len(X_test), len(y_test))
def split_long_texts(text_tokens, labels):

    _text_tokens = []

    _labels = []

    for text, label in zip(text_tokens, labels):

        text_size = len(text) // max_length

        text_res = len(text) % max_length



        for i in range(text_size):

            _text_tokens.append(text[i: i + max_length])

            _labels.append(label)



        if text_res > 0.5 * max_length or text_size == 0:

            _text_tokens.append(text[text_size * max_length:])

            _labels.append(label)

    return _text_tokens, np.array(_labels)
tokenizer_obj = Tokenizer()

total_reviews = np.hstack((X_train, X_test))

tokenizer_obj.fit_on_texts(total_reviews)



vocab_size = len(tokenizer_obj.word_index) + 1



X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)

X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)



X_train_tokens, y_train = split_long_texts(X_train_tokens, y_train)

X_test_tokens, y_test = split_long_texts(X_test_tokens, y_test)



X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')

X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')
X_train_pad.shape
vocab_size
#vocab_size = 124253
max_length
EMBEDDING_DIM = 2



print('Build model...')



model = Sequential()

model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))

model.add(BatchNormalization())

model.add(Dropout(0.2))

#model.add(GRU(units=1, return_sequences=True)) #, dropout=0.3, recurrent_dropout=0.3))

model.add(LSTM(2, dropout=0.1, recurrent_dropout=0.1))

#Global Maxpooling

model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))



optimizer = Adam(lr=0.0005)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
print('Train...')



my_callbacks = [

    EarlyStopping(patience=10),

]



model.fit(X_train_pad, y_train, batch_size=128, epochs=5, validation_data=(X_test_pad, y_test), verbose=2, callbacks=my_callbacks)
model.save('model.h5')
model.save_weights('model_weights.h5')
model_weights
os.listdir('../input/learning-word-vectors-for-sentiment-analysis/')
model.load_weights(model_weights)
import pickle



# saving

with open('tokenizer.pickle', 'wb') as handle:

    pickle.dump(tokenizer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)



# loading

with open('tokenizer.pickle', 'rb') as handle:

    tokenizer = pickle.load(handle)
model = keras.models.load_model('../working/model.h5')
test_sample_1 = 'This movie is fantasric! I realy like it because it is so good!'

test_sample_2 = 'Good movie!'

test_sample_3 = 'Bad movie!'

test_samples = [test_sample_1, test_sample_2, test_sample_3]



test_samples_tokens = tokenizer.texts_to_sequences(test_samples)

test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)



model.predict(x=test_samples_tokens_pad)
from scipy.stats import norm



# https://www.quora.com/Do-the-user-ratings-on-IMDB-follow-a-bell-curve-If-so-what-is-the-mean-and-standard-deviation

mean = 6.2

std = 1.4 



norm.pdf(10, loc=mean, scale=std)
rate_dict = {i: 0 for i in range(1, 11)}

prev_prob = 0

for i, rate in enumerate([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10], 1):

    curr_prob = norm.cdf(rate, loc=mean, scale=std)

    rate_dict[i] = [prev_prob, curr_prob]

    prev_prob = curr_prob

rate_dict[10] = [rate_dict[10][0], 1]

print(rate_dict)
def get_score(p, rate_dict):

    for key, interval in rate_dict.items():

        if interval[0] < p < interval[1]:

            return key
for p in model.predict(x=test_samples_tokens_pad).reshape(-1).tolist():

    print(p, get_score(p, rate_dict))
prediction = model.predict(x=X_test_pad)
roc_auc_score(y_test, prediction.reshape(-1))
%%time



review_lines = list()

lines = df['text'].values.tolist()



stop_words = set(stopwords.words('english'))



for line in lines:

    tokens = word_tokenize(line)

    tokens = [w.lower() for w in tokens]

    table = str.maketrans('', '', string.punctuation)

    stripped = [w.translate(table) for w in tokens]

    words = [word for word in stripped if word.isalpha()]

    words = [w for w in words if not w in stop_words]

    review_lines.append(words)
len(review_lines)
%%time

EMBEDDING_DIM = 256



model = gensim.models.Word2Vec(sentences=review_lines, size=EMBEDDING_DIM, window=5, workers=4, min_count=5)

words = list(model.wv.vocab)

print('Vocabulary size: %d' % len(words))
## Vocabulary size: 28115
model.wv.most_similar('horrible')
filename = 'imdb_embedding_word2vec.txt'

model.wv.save_word2vec_format(filename, binary=False)
embedding_index = {}

with open(os.path.join('./imdb_embedding_word2vec.txt')) as fin:

    for line in fin:

        values = line.split()

        if len(values) == 2:

            print('Num words - ', values[0])

            print('EMBEDDING_DIM =', values[1])

            continue

        word = values[0]

        coefs = np.asarray(values[1:])

        embedding_index[word] = coefs
len(embedding_index.keys())
tokenizer_obj = Tokenizer()

total_reviews = df_labeled['text'].values

tokenizer_obj.fit_on_texts(total_reviews)

sequences = tokenizer_obj.texts_to_sequences(total_reviews)



word_index = tokenizer_obj.word_index

print('Found %s unique tokens.' % len(word_index))



review_pad = pad_sequences(sequences, maxlen=max_length, padding='post')

sentiment = df_labeled['sentiment'].values

print(review_pad.shape)

print(sentiment.shape)
word_index
num_words = len(word_index) + 1

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))



words_n = 0

finde_n = 0

for word, i in word_index.items():

    words_n += 1

    if i > num_words:

        continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None:

        finde_n += 1

        embedding_matrix[i] = embedding_vector
rev_model = Sequential()

embedding_layer = Embedding(

    num_words,

    EMBEDDING_DIM,

    embeddings_initializer=Constant(embedding_matrix),

    input_length=max_length,

    trainable=False,

)



rev_model.add(embedding_layer)

rev_model.add(BatchNormalization())

rev_model.add(Dropout(0.2))

#rev_model.add(LSTM(64)) #, return_sequences=True)) # , dropout=0.2, recurrent_dropout=0.2))

rev_model.add(GRU(units=32))#, dropout=0.1, recurrent_dropout=0.1))

rev_model.add(BatchNormalization())

rev_model.add(Dropout(0.2))

rev_model.add(Dense(1, activation='sigmoid'))



optimizer = Adam(lr=0.0005)

rev_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
rev_model.summary()
VALIDATION_SPLIT = 0.2



indices = np.arange(review_pad.shape[0])

np.random.shuffle(indices)

review_pad = review_pad[indices]

sentiment = sentiment[indices]

num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])



X_train_pad = review_pad[:-num_validation_samples]

y_train = sentiment[:-num_validation_samples]

X_test_pad = review_pad[-num_validation_samples:]

y_test = sentiment[-num_validation_samples:]
print(X_train_pad.shape)

print(y_train.shape)

print(X_test_pad.shape)

print(y_test.shape)
print('Train...')



my_callbacks = [

    EarlyStopping(patience=10),

]

rev_model.fit(X_train_pad, y_train, batch_size=128, epochs=100, validation_data=(X_test_pad, y_test), verbose=2, callbacks=my_callbacks)
rev_model.evaluate(X_test_pad, y_test)
rev_model.evaluate(X_train_pad, y_train)
rev_model.save('rev_model.h5')
rev_model.save_weights('rev_model_weights.h5')
prediction = rev_model.predict(x=X_test_pad)
roc_auc_score(y_test, prediction.reshape(-1))