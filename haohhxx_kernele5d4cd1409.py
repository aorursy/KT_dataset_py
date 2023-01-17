import pickle
import platform
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import (Input, Dense, Embedding, Bidirectional,
                          Conv1D, Dropout, BatchNormalization, Activation, CuDNNGRU, CuDNNLSTM, Multiply, Layer)
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D, SpatialDropout2D

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD, Nadam
from keras.models import Model
from keras import backend as K
from keras import initializers

__system__ = platform.system()

# MAX_LEN = 100
MAX_LEN = 50
EMBEDDING_DIM = 300
MAX_FEATURES = 100000
RANDOM_STATE = 91

if __system__ is 'Windows':
    GLOVE_DIR = "D:/data/word2vec/en/crawl-300d-2M.vec/crawl-300d-2M.vec"
else:
    GLOVE_DIR = "/mnt/haohhxx/d/data/word2vec/en/crawl-300d-2M.vec/crawl-300d-2M.vec"
    
GLOVE_DIR = r"../input/fatsttext-common-crawl/crawl-300d-2M/crawl-300d-2M.vec"


def preprocessing(train, test, max_len=MAX_LEN, max_features=MAX_FEATURES, train_size=0.75):
    """
        https://www.kaggle.com/antmarakis/cnn-baseline-model
    """
    X = train['Phrase'].values.tolist()
    # X = [x.replace("n't", "not") for x in X]
    X_test = test['Phrase'].values.tolist()
    # X_test = [x.replace("n't", "not") for x in X_test]

    X_tok = X + X_test
    tokenizer = Tokenizer(num_words=max_features, filters='')
    tokenizer.fit_on_texts(X_tok)

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=max_len)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_test = pad_sequences(X_test, maxlen=max_len)

    word_index = tokenizer.word_index

    y = train['Sentiment'].values

    Y = to_categorical(y)
    X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                          Y,
                                                          train_size=train_size,
                                                          shuffle=True,
                                                          random_state=RANDOM_STATE,
                                                          stratify=y)

    loss_weights = [1 / 5 for _ in range(5)]
    return X_train, X_valid, y_train, y_valid, X_test, loss_weights, word_index


def get_model(embedding_matrix, max_len=MAX_LEN, units=128, dr=0.5, embed_size=EMBEDDING_DIM):
    inp = Input(shape=(max_len,))
    x = Embedding(19479, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x1 = SpatialDropout1D(dr)(x)

    x_gru = Bidirectional(CuDNNGRU(units, return_sequences=True))(x1)
    x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x1)
    max_pool1_gru = GlobalMaxPooling1D()(x1)

    x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool3_gru = GlobalAveragePooling1D()(x3)
    max_pool3_gru = GlobalMaxPooling1D()(x3)

    x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x1)
    x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = GlobalAveragePooling1D()(x1)
    max_pool1_lstm = GlobalMaxPooling1D()(x1)

    x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool3_lstm = GlobalAveragePooling1D()(x3)
    max_pool3_lstm = GlobalMaxPooling1D()(x3)

    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
                     avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(128, activation='relu')(x))
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(100, activation='relu')(x))
    out = Dense(5, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)

    return model


def get_model_t(embedding_matrix, word_index, max_len=MAX_LEN, units=128, dr=0.3, embed_size=300):
    inp = Input(shape=(max_len,))
    x = Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x1 = SpatialDropout1D(dr)(x)
    # x1 = Dropout(dr)(x)
    x1 = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x1)

    conv2 = Conv1D(128, kernel_size=2, padding='valid', kernel_initializer='he_uniform', activation='relu')(x1)
    conv3 = Conv1D(128, kernel_size=3, padding='valid', kernel_initializer='he_uniform', activation='relu')(x1)

    x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x1)
    x_lstm_c2 = Bidirectional(CuDNNLSTM(units, return_sequences=True))(conv2)
    x_lstm_c3 = Bidirectional(CuDNNLSTM(units, return_sequences=True))(conv3)
    # batch * seqlen * 2hidden

    atl1 = Dense(24, activation='relu', input_shape=(50,))(x_lstm)
    atl2 = Dense(1, activation='softmax', input_shape=(50,))(atl1)
    x_lstm = Multiply()([x_lstm, atl2])
    # atl1 = Dense(24, activation='relu', input_shape=(50,))(x_lstm_c2)
    # atl2 = Dense(1, activation='softmax', input_shape=(50,))(atl1)
    # x_lstm_c2_at = Multiply()([x_lstm_c2, atl2])
    # atl1 = Dense(24, activation='relu', input_shape=(50,))(x_lstm_c3)
    # atl2 = Dense(1, activation='softmax', input_shape=(50,))(atl1)
    # x_lstm_c3_at = Multiply()([x_lstm_c3, atl2])

    # avg_pool1_lstm = GlobalAveragePooling1D()(x_lstm)
    max_pool1_lstm = GlobalMaxPooling1D()(x_lstm)
    max_pool1_lstm_c2 = GlobalMaxPooling1D()(x_lstm_c2)
    max_pool1_lstm_c3 = GlobalMaxPooling1D()(x_lstm_c3)

    # max_pool1_conv2 = GlobalMaxPooling1D()(conv2)
    # max_pool1_conv3 = GlobalMaxPooling1D()(conv3)
    # max_pool1_conv4 = GlobalMaxPooling1D()(conv4)
    # max_pool1_conv6 = GlobalMaxPooling1D()(conv6)
    convs = concatenate([max_pool1_lstm_c2, max_pool1_lstm_c3, max_pool1_lstm])

    x = BatchNormalization()(convs)
    x = Dropout(0.1)(Dense(256, activation='relu')(x))
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(100, activation='relu')(x))
    out = Dense(5, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)

    return model

#
# def attention_layer(X, n_h, Ty):
#     """
#     Creates an attention layer.
#
#     Input:
#     X - Layer input (m, Tx, x_vocab_size)
#     n_h - Size of LSTM hidden layer
#     Ty - Timesteps in output sequence
#
#     Output:
#     output - The output of the attention layer (m, Tx, n_h)
#     """
#     # Define the default state for the LSTM layer
#     h = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], n_h)))(X)
#     c = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], n_h)))(X)
#     # Messy, but the alternative is using more Input()
#
#     at_LSTM = LSTM(n_h, return_state=True)
#
#     output = []
#
#     # Run attention step and RNN for each output time step
#     for _ in range(Ty):
#         context = one_step_of_attention(h, X)
#
#         h, _, c = at_LSTM(context, initial_state=[h, c])
#
#         output.append(h)
#
#     return output
#
#
# def model_lstm(embedding_matrix, word_index, max_len=MAX_LEN, units=128, dr=0.3, embed_size=300):
#     inp = Input(shape=(max_len,))
#     x = Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix], trainable=True)(inp)
#     x1 = SpatialDropout1D(dr)(x)
#     x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences=True))(x1)
#
#     atl1 = Dense(24, activation='relu', input_shape=(50,))(x_lstm)
#     atl2 = Dense(1, activation='softmax', input_shape=(50,))(atl1)
#     x_lstm = Multiply()([x_lstm, atl2])
#
#     max_pool1_lstm = GlobalMaxPooling1D()(x_lstm)
#
#     x = BatchNormalization()(max_pool1_lstm)
#     x = Dropout(0.1)(Dense(128, activation='relu')(x))
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(Dense(100, activation='relu')(x))
#     out = Dense(5, activation="softmax")(x)
#
#     model = Model(inputs=inp, outputs=out)
#
#     return model
#

def get_glove(word_index, path=GLOVE_DIR):
    """
        https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    """
    embeddings_index = {}
    with open(path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            values = line.strip().split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def prepare(train_size):
    train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', sep="\t")
    test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', sep="\t")

    X_train, X_valid, y_train, y_valid, X_test, loss_weights, word_index = preprocessing(train, test,
                                                                                         train_size=train_size)
    data = {
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "X_test": X_test,
        "loss_weights": loss_weights,
        "word_index": word_index,
    }
    pickle.dump(data, open("./keras.data", 'wb'))


def main():
    train_size = 0.9
    prepare(train_size)

    data = pickle.load(open("./keras.data", 'rb'))
    X_train = data["X_train"]
    X_valid = data["X_valid"]
    y_train = data["y_train"]
    y_valid = data["y_valid"]
    X_test = data["X_test"]
    loss_weights = data["loss_weights"]
    word_index = data["word_index"]
    glove_emb = get_glove(word_index)
    # model = get_model(glove_emb)
    model = get_model_t(glove_emb, word_index)
    model.summary()

    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-4, decay=0), metrics=["accuracy"])
    # opt = Nadam(lr=1e-4, schedule_decay=0.04, decay=0)
    # model.compile(loss=weighted_categorical_crossentropy(loss_weights),
    #               optimizer=opt,
    #               metrics=['accuracy'])

    check_point = ModelCheckpoint('best_weights.h5',
                                  monitor='val_acc',
                                  verbose=0,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='max',
                                  period=1)

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)

    # model.fit(X_train, y_ohe, batch_size=128, epochs=15, validation_split=0.1, verbose=1,
    #           callbacks=[check_point, early_stop])
    model.fit(X_train,
              y_train,
              batch_size=128,
              epochs=20,
              verbose=1,
              validation_data=[X_valid, y_valid],
              callbacks=[check_point, early_stop])
    model.load_weights('best_weights.h5')
    sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')

    sub['Sentiment'] = np.argmax(model.predict(X_test, verbose=1), axis=-1)
    sub.to_csv('sub.csv', index=False)


if __name__ == '__main__':
    main()

