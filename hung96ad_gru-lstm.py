import numpy as np

import pandas as pd

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate

from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.optimizers import Adam

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler
EMBEDDING_FILES = [

    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]

NUM_MODELS = 2

MAX_FEATURES = 100000

BATCH_SIZE = 512*3

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 1

MAX_LEN = 220

def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')
def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)
def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix
def build_model(embedding_matrix, num_aux_targets):

    words = Input(shape=(MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.3)(x)

    x2 = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x2 = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x2)

    x1 = Bidirectional(CuDNNGRU(LSTM_UNITS, return_sequences=True))(x)

    x1 = Bidirectional(CuDNNGRU(LSTM_UNITS, return_sequences=True))(x1)

    x = add([x1, x2])





    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(clipnorm=0.1),

        metrics=['accuracy'])



    return model
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
x_train = train['comment_text'].fillna('').values

y_train = np.where(train['target'] >= 0.5, 1, 0)

y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

x_test = test['comment_text'].fillna('').values
tokenizer = text.Tokenizer(num_words=MAX_FEATURES)

tokenizer.fit_on_texts(list(x_train) + list(x_test))
x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
embedding_matrix = np.concatenate(

    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
checkpoint_predictions = []

weights = []
for model_idx in range(NUM_MODELS):

    model = build_model(embedding_matrix, y_aux_train.shape[-1])

    for global_epoch in range(EPOCHS):

        model.fit(

            x_train,

            [y_train, y_aux_train],

            batch_size=BATCH_SIZE,

            epochs=1,

            callbacks=[

                LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))

            ]

        )

        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

        weights.append(2 ** global_epoch)
predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
submission = pd.DataFrame.from_dict({

    'id': test['id'],

    'prediction': predictions

})

submission.to_csv('submission.csv', index=False)