%%time

import numpy as np

import pandas as pd

from sklearn import preprocessing, model_selection

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, Dropout, BatchNormalization, Conv1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler
%%time

DATA_FILE = '../input/songlyrics/songdata.csv'

EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'

TEXT_COLUMNS = 'text'

TARGET_COLUMNS = 'artist'



EPOCHS = 10

BATCH_SIZE = 256

LSTM_UNITS = 128

DENSE_UNITS = 4 * LSTM_UNITS

MAX_LEN = 1000

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))



    for word, i in word_index.items():

        

        embedding_vector = embedding_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

            continue

            

        embedding_vector = embedding_index.get(word.lower())

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

            continue

            

        embedding_vector = embedding_index.get(word.upper())

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

            continue

            

        embedding_vector = embedding_index.get(word.title())

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

            continue

            

        embedding_matrix[i] = np.random.normal(loc=0, scale=1, size=(1,300))

        

    return embedding_matrix



def build_model(embedding_matrix, out_size):

    words = Input(shape=(None,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = GlobalMaxPooling1D()(x)

    hidden = Dense(DENSE_UNITS, activation='relu')(hidden)

    hidden = Dropout(0.2)(hidden)

    hidden = BatchNormalization()(hidden)

    result = Dense(out_size, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=result)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    return model
%%time

df = pd.read_csv(DATA_FILE, usecols=[TARGET_COLUMNS, TEXT_COLUMNS], dtype=str)

df = pd.merge(df, df.groupby(TARGET_COLUMNS).size().to_frame(name='size') > 180, how='left', on=TARGET_COLUMNS)

df = df[df['size']][[TARGET_COLUMNS, TEXT_COLUMNS]]

print('data size:', df.shape)



n_class = df[TARGET_COLUMNS].unique().shape[0]

print('n_class: ', n_class)



df[TARGET_COLUMNS] = preprocessing.LabelEncoder().fit_transform(df[TARGET_COLUMNS])

df = df.sample(frac=1) # shuffle



X = df[TEXT_COLUMNS].astype(str)

y = df[TARGET_COLUMNS].values



tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(X)



X = tokenizer.texts_to_sequences(X)

X = sequence.pad_sequences(X, maxlen=MAX_LEN)



y = preprocessing.LabelBinarizer(neg_label=0, pos_label=1).fit_transform(y)



X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.3)

print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)



embedding_matrix = build_matrix(tokenizer.word_index, EMBEDDING_FILE)
%%time

model = build_model(embedding_matrix, n_class)

for global_epoch in range(EPOCHS):

    model.fit(

        X_train, y_train, batch_size=BATCH_SIZE, epochs=1, verbose=2, validation_data=(X_valid, y_valid),

        callbacks=[LearningRateScheduler(lambda _: 1e-3 * (0.9 ** global_epoch))]

    )
def batch_iter(X, y, batch_size, shuffle=True):

    num_batches_per_epoch = int((len(X) - 1) / batch_size) + 1



    def data_generator(X, y, batch_size, shuffle):

        data_size = len(X)

        while True:

            # Shuffle the data at each epoch

            if shuffle:

                shuffle_indices = np.random.permutation(np.arange(data_size))

                shuffled_X = X[shuffle_indices]

                shuffled_y = y[shuffle_indices]

            else:

                shuffled_X = X

                shuffled_y = y



            for batch_num in range(num_batches_per_epoch):

                start_index = batch_num * batch_size

                end_index = min((batch_num + 1) * batch_size, data_size)

                X = shuffled_X[start_index: end_index]

                X = X[:, -int(np.percentile(np.sum(X != 0, axis=1), 95)):]

                y = shuffled_y[start_index: end_index]

                yield X, y



    return num_batches_per_epoch, data_generator(X, y, batch_size, shuffle)
%%time

model = build_model(embedding_matrix, n_class)

for global_epoch in range(EPOCHS):

    train_steps, train_batches = batch_iter(X_train, y_train, BATCH_SIZE)

    valid_steps, valid_batches = batch_iter(X_valid, y_valid, BATCH_SIZE)

    model.fit_generator(

        train_batches, train_steps, epochs=1, verbose=2,

        validation_data=valid_batches, validation_steps=valid_steps,

        callbacks=[LearningRateScheduler(lambda _: 1e-3 * (0.9 ** global_epoch))]

    )