import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import numpy as np 

import pandas as pd 

import os

import gc

import logging

import datetime

import warnings

import pickle

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.compat.v1.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from tensorflow.compat.v1.keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate

from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from tensorflow.compat.v1.keras.preprocessing import text, sequence

from tensorflow.compat.v1.keras.losses import binary_crossentropy

from tensorflow.compat.v1.keras import backend as K

import tensorflow.compat.v1.keras.layers as L

from tensorflow.compat.v1.keras import initializers, regularizers, constraints, optimizers, layers



from tensorflow.compat.v1.keras.models import Model

from tensorflow.compat.v1.keras.optimizers import Adam

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from tensorflow.compat.v1.keras.preprocessing.text import Tokenizer

from tensorflow.compat.v1.keras.preprocessing.sequence import pad_sequences
COMMENT_TEXT_COL = 'comment_text'

EMB_MAX_FEAT = 300

MAX_LEN = 220

BATCH_SIZE = 512

NUM_EPOCHS = 4

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 512

NUM_MODELS = 2

EMB_PATHS = [

    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]

JIGSAW_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/'
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)
def custom_loss(y_true, y_pred):

    return binary_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_pred) * y_true[:,1]
def run_proc_and_tokenizer(train, test):

    '''

        credits go to: https://www.kaggle.com/tanreinama/simple-lstm-using-identity-parameters-solution/ 

    '''

 

    identity_columns = ['asian', 'atheist',

       'bisexual', 'black', 'buddhist', 'christian', 'female',

       'heterosexual', 'hindu', 'homosexual_gay_or_lesbian',

       'intellectual_or_learning_disability', 'jewish', 'latino', 'male',

       'muslim', 'other_disability', 'other_gender',

       'other_race_or_ethnicity', 'other_religion',

       'other_sexual_orientation', 'physical_disability',

       'psychiatric_or_mental_illness', 'transgender', 'white']

       

    # Overall

    weights = np.ones((len(train),)) / 4

    # Subgroup

    weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4

    # Background Positive, Subgroup Negative

    weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +

       (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

    # Background Negative, Subgroup Positive

    weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +

       (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

    loss_weight = 1.0 / weights.mean()

    

    y_train = np.vstack([(train['target'].values>=0.5).astype(np.int),weights]).T

    y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values

    

    tokenizer = Tokenizer() 

    tokenizer.fit_on_texts(list(train[COMMENT_TEXT_COL]) + list(test[COMMENT_TEXT_COL]))

    word_index = tokenizer.word_index

    X_train = tokenizer.texts_to_sequences(list(train[COMMENT_TEXT_COL]))

    X_test = tokenizer.texts_to_sequences(list(test[COMMENT_TEXT_COL]))

    X_train = pad_sequences(X_train, maxlen=MAX_LEN)

    X_test = pad_sequences(X_test, maxlen=MAX_LEN)



    del identity_columns, weights, tokenizer, train, test

    gc.collect()

    

    return X_train, y_train, X_test, y_aux_train, word_index, loss_weight
def build_embedding_matrix(word_index, path):

    '''

     credits to: https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold

    '''

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, EMB_MAX_FEAT))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

        except:

            embedding_matrix[i] = embeddings_index["unknown"]

            

    del embedding_index

    gc.collect()

    return embedding_matrix
def build_embeddings(word_index):

    embedding_matrix = np.concatenate(

        [build_embedding_matrix(word_index, f) for f in EMB_PATHS], axis=-1) 

    return embedding_matrix
# function to get train , test dataset and pre-trained embeddings

def load_data():

    train = pd.read_csv(os.path.join(JIGSAW_PATH,'train.csv'), index_col='id')

    test = pd.read_csv(os.path.join(JIGSAW_PATH,'test.csv'), index_col='id')

    y_train = np.where(train['target'] >= 0.5, True, False) * 1

    X_train, y_train, X_test, y_aux_train, word_index, loss_weight = run_proc_and_tokenizer(train, test)

    embedding_matrix = build_embeddings(word_index)

    del train,test

    gc.collect()

    return X_train, y_train, X_test, y_aux_train, word_index, embedding_matrix, loss_weight
def build_model(embedding_matrix, num_aux_targets, loss_weight):

    words = Input(shape=(MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)



    hidden = concatenate([GlobalMaxPooling1D()(x),GlobalAveragePooling1D()(x),])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss=[custom_loss,'binary_crossentropy'], loss_weights=[loss_weight, 1.0], 

                  optimizer='adam')

    return model
def run_model(X_train, y_train,X_test, y_aux_train, embedding_matrix, word_index, loss_weight):

  

    checkpoint_predictions = []

    weights = []



    model = build_model(embedding_matrix, y_aux_train.shape[-1], loss_weight)

    file_path = "best_model.h5"

    tensorboard_callback = TensorBoard("logs")

    model.fit(

        X_train, [y_train, y_aux_train],

        batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1,

        callbacks=[LearningRateScheduler(lambda epoch: 1.1e-3 * (0.55 ** epoch)),tensorboard_callback]

    )

    del X_train,y_train,y_aux_train,embedding_matrix

    gc.collect()

    

    predictions =model.predict(X_test, batch_size=2048)[0].flatten()

    model.save(file_path)

    del model, X_test

    gc.collect()

    

    return predictions
def submit(sub_preds):

    submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')

    submission['prediction'] = sub_preds

    submission.reset_index(drop=False, inplace=True)

    submission.to_csv('submission.csv', index=False)
def main():

    X_train, y_train, X_test, y_aux_train, word_index,embedding_matrix, loss_weight = load_data()

    model = build_model(embedding_matrix, y_aux_train.shape[-1], loss_weight)

    model.summary()

    del model

    gc.collect()

    sub_preds = run_model(X_train, y_train, X_test, y_aux_train, embedding_matrix, word_index, loss_weight)

    submit(sub_preds)

    

if __name__ == "__main__":

    main()