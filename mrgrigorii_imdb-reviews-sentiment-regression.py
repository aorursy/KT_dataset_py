# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import gensim



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



from sklearn.metrics import roc_auc_score



from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint



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



%matplotlib inline
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
df = pd.read_csv('../input/imdb-reviews-dataset/imdb_reviews_dataset.csv')

df.head()
df['rating'].unique()
df_labeled = df[df['rating'] != 0].copy() 

df_labeled.shape
rating_count = df_labeled.groupby('rating').count()

plt.bar(rating_count.index, rating_count.text)

plt.xlabel('Rating')

plt.ylabel('Count')
df_labeled['sentiment'] = df_labeled['rating'].apply(lambda x: 1 if x >= 7 else 0)
df_labeled['len_text'] = df_labeled['text'].apply(lambda x: len(x.split()))

df_labeled.head()
len_text_info = df_labeled['len_text'].describe()

len_text_info
# set max len for padding

max_length = int(len_text_info['mean'] + 2 * len_text_info['std'])

#max_length = int(len_text_info['mean'])

print(max_length) # = 200
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

EMBEDDING_DIM = 512



model = gensim.models.Word2Vec(sentences=review_lines, size=EMBEDDING_DIM, window=5, workers=4, min_count=5)

words = list(model.wv.vocab)

print('Vocabulary size: %d' % len(words))
model.wv.most_similar('horrible')
filename = 'imdb_embedding_word2vec.txt'

model.wv.save_word2vec_format(filename, binary=False)
embedding_index = {}

with open(filename) as fin:

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

rating = df_labeled['rating'].values

print(review_pad.shape)

print(sentiment.shape)
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

#rev_model.add(BatchNormalization())

rev_model.add(Dropout(0.2))

#rev_model.add(LSTM(128)) #, return_sequences=True)) # , dropout=0.2, recurrent_dropout=0.2))

rev_model.add(GRU(units=32))#, dropout=0.1, recurrent_dropout=0.1))

#rev_model.add(BatchNormalization())

rev_model.add(Dropout(0.2))

rev_model.add(Dense(1, activation='relu'))



optimizer = Adam(lr=0.0005)

rev_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
rev_model.summary()
VALIDATION_SPLIT = 0.2



indices = np.arange(review_pad.shape[0])

np.random.shuffle(indices)

review_pad = review_pad[indices]

sentiment = sentiment[indices]

rating = rating[indices]

num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])



X_train_pad = review_pad[:-num_validation_samples]

y_train = sentiment[:-num_validation_samples]

y_train_r = rating[:-num_validation_samples] / 10.

X_test_pad = review_pad[-num_validation_samples:]

y_test = sentiment[-num_validation_samples:]

y_test_r = rating[-num_validation_samples:] / 10.
print(X_train_pad.shape)

print(y_train_r.shape)

print(X_test_pad.shape)

print(y_test_r.shape)
print('Train...')



my_callbacks = [

    EarlyStopping(patience=50),

    ModelCheckpoint(filepath='model_best.h5', save_best_only=True),

]

rev_model.fit(X_train_pad, y_train_r, batch_size=128, epochs=500, validation_data=(X_test_pad, y_test_r), verbose=2, callbacks=my_callbacks)
rev_model.save('rev_model.h5')
rev_model.save_weights('rev_model_weights.h5')
model = keras.models.load_model('model_best.h5')
eval_result = model.evaluate(X_test_pad, y_test_r)

eval_result
ratings = [1, 2, 3, 4, 7, 8, 9, 10]

ratings_mae = []



for rating in ratings:

    print('Rating - ', rating)

    idxs = np.where(y_test_r * 10 == rating)

    eval_result = model.evaluate(X_test_pad[idxs], y_test_r[idxs])

    ratings_mae.append(eval_result[0])
plt.bar(ratings, ratings_mae)
prediction = model.predict(x=X_test_pad)
roc_auc_score(y_test, prediction.reshape(-1))
import pickle



# saving

with open('tokenizer_word2vec.pickle', 'wb') as handle:

    pickle.dump(tokenizer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
# saving

with open('embedding_matrix.pickle', 'wb') as handle:

    pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)