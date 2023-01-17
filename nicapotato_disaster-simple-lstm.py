# Adapted from thousandvoices - https://www.kaggle.com/thousandvoices/simple-lstm/code

# Minor changes to LSTM layer convention



import numpy as np

import pandas as pd

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from gensim.models import KeyedVectors



import keras

print(keras.__version__)

import tensorflow

print(tensorflow.__version__)



EMBEDDING_FILES = [

    '../input/gensim-embeddings-dataset/crawl-300d-2M.gensim',

    '../input/gensim-embeddings-dataset/glove.840B.300d.gensim'

]



N_ROWS = None

NUM_MODELS = 3

BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4

LOCAL_EPOCHS = 2

MAX_LEN = 220

AUX_COLUMNS = ['target']

TEXT_COLUMN = 'text'

TARGET_COLUMN = 'target'

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
def build_matrix(word_index, path):

    embedding_index = KeyedVectors.load(path, mmap='r')

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        for candidate in [word, word.lower()]:

            if candidate in embedding_index:

                embedding_matrix[i] = embedding_index[candidate]

                break

    return embedding_matrix

    

def build_model(embedding_matrix, num_aux_targets):

    words = Input(shape=(None,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model
print("Read Data")

train_df = pd.read_csv('../input/nlp-getting-started/train.csv', nrows = N_ROWS)

test_df = pd.read_csv('../input/nlp-getting-started/test.csv', nrows = N_ROWS)



x_train = train_df[TEXT_COLUMN].astype(str)

y_train = train_df[TARGET_COLUMN].values

y_aux_train = train_df[AUX_COLUMNS].values

x_test = test_df[TEXT_COLUMN].astype(str)



tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE, lower=False)

tokenizer.fit_on_texts(list(x_train) + list(x_test))



x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)



embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)



checkpoint_predictions = []

weights = []



model = build_model(embedding_matrix, y_aux_train.shape[-1])
print("Start Modeling")

for model_idx in range(1, NUM_MODELS + 1):

    print("Start Model: {}".format(str(model_idx)))

    model = build_model(embedding_matrix, y_aux_train.shape[-1])

    for global_epoch in range(EPOCHS):

        model.fit(

            x_train,

            [y_train, y_aux_train],

            batch_size=BATCH_SIZE,

            epochs=LOCAL_EPOCHS,

            verbose=1

        )

        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

        weights.append(2 ** global_epoch)

print("Modeling Complete")
predictions = np.average(checkpoint_predictions, weights=weights, axis=0)



submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    TARGET_COLUMN: (predictions >.5).astype(int)

})

submission.to_csv('lstm_submission.csv', index=False)

submission.head()