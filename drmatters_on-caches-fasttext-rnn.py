!mkdir -p ../working/cache/vectorized
import gc

import io

import os

import pickle

from datetime import datetime

import matplotlib.pyplot as plt



import nltk

import numpy as np

import pandas as pd

from tqdm import tqdm





class InferenceOnTexts:

    def __init__(self, predict, aggregate=max):

        self.predict = predict

        self.aggregate = aggregate



    def run(self, texts: pd.Series):

        res = []

        print('Start inference on texts')

        for document in tqdm(texts):

            sentences = nltk.sent_tokenize(document, language='russian')

            sentences_preds = self.predict(sentences)

            aggregated = self.aggregate(sentences_preds)

            res.append(aggregated)

        res = np.asarray(res)

        gc.collect()

        return res





def load_dataset(folder: str, load_test=True, info=True):

    core: pd.DataFrame = pd.read_csv(folder + '/core.csv')

    if load_test:

        test: pd.DataFrame = pd.read_csv(folder + '/test_data.csv')



    if info:

        if load_test:

            print(test.info())

        print(core.info())



    print('Dataset loaded')

    print('______________')



    if load_test:

        return core, test

    else:

        return core





def save_inference_res(test: pd.DataFrame, inf_result: np.ndarray, run_name: str,

                       folder: str):

    res = test

    res['pred'] = inf_result

    res = res.sort_values('pred', ascending=False)

    time_str = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')

    name = f'({time_str}){run_name}.csv'

    path = os.path.join(folder, name)

    res.to_csv(path, index=False)





def plot_graphs(history, string):

    plt.plot(history.history[string])

    plt.plot(history.history['val_'+string])

    plt.xlabel("Epochs")

    plt.ylabel(string)

    plt.legend([string, 'val_'+string])

    plt.show()
import functools

import random

import pathlib



import nltk

import numpy as np

from nltk.corpus import stopwords

from sklearn import model_selection

from tqdm import tqdm
RANDOM_SEED = 42

SEQ_LEN = 55

TEST_SIZE = 0.2



CACHE_FOLDER = pathlib.Path('../input/fasttext-p-rnn-caches/cache')

CACHE_OUTPUT_FOLDER = pathlib.Path('../working/cache')

MAKE_SUBMISSION = False

MAKE_PROBA = True



EPOCHS = 100

BATCH = 128



random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)
core = load_dataset('../input/nlp-task/', load_test=False, info=False)

train, val = model_selection.train_test_split(core, test_size=TEST_SIZE,

                                              random_state=RANDOM_SEED,

                                              stratify=core['target'])

y_train, y_val = train['target'].values, val['target'].values





in_path_folder = CACHE_FOLDER / 'vectorized/'

print("Reading cached vectorized representations")

train_vectorized = pickle.load(open(in_path_folder / 'train_vectorized.pkl', 'rb'))

val_vectorized = pickle.load(open(in_path_folder / 'val_vectorized.pkl', 'rb'))

test_titles_vectorized = pickle.load(open(in_path_folder / 'test_titles_vectorized.pkl', 'rb'))

test_text_vectorized = pickle.load(open(in_path_folder / 'test_text_vectorized.pkl', 'rb'))

print("Completed!")
print("Reading cached embedding matrix")

emb_mat = np.load(os.path.join(CACHE_FOLDER, 'emb_matrix.npy'))

np.save(os.path.join(CACHE_OUTPUT_FOLDER, 'emb_matrix.npy'), emb_mat)

print("Completed!")
import tensorflow as tf

tf.random.set_seed(RANDOM_SEED)

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Input, GRU, LSTM, Embedding, Activation

from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D, Concatenate, SpatialDropout1D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras import backend as K

from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
def recall_fn(y_true, y_pred):

    """Recall metric.



    Only computes a batch-wise average of recall.



    Computes the recall, a metric for multi-label classification of

    how many relevant items are selected.

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_fn(y_true, y_pred):

    """Precision metric.



    Only computes a batch-wise average of precision.



    Computes the precision, a metric for multi-label classification of

    how many selected items are relevant.

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1(y_true, y_pred):

    precision = precision_fn(y_true, y_pred)

    recall = recall_fn(y_true, y_pred)

    return 2*((precision * recall)/(precision + recall + K.epsilon()))



def build_model_third_place(inp_len, embedding_matrix, learning_rate=0.001):

    inp = Input(shape=(inp_len,))

    x = Embedding(embedding_matrix.shape[0],

                  embedding_matrix.shape[1],

                  weights=[embedding_matrix],

                  trainable=False)(inp)

    x = SpatialDropout1D(0.3)(x)

    x1 = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True))(x)

    x2 = Bidirectional(GRU(32, dropout=0.5, return_sequences=True))(x1)

    max_pool1 = GlobalMaxPooling1D()(x1)

    max_pool2 = GlobalMaxPooling1D()(x2)

    conc = Concatenate()([max_pool1, max_pool2])

    predictions = Dense(1, activation='sigmoid')(conc)

    

    model = Model(inputs=inp, outputs=predictions)

    adam = optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[f1])

    return model



def build_model_simple_pavel(inp_len, embedding_matrix, learning_rate=0.001):

    model=Sequential()



    embedding=Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],

                        embeddings_initializer=initializers.Constant(embedding_matrix),

                        input_length=inp_len,trainable=False)



    model.add(embedding)

    model.add(SpatialDropout1D(0.2))

    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(1, activation='sigmoid'))



    optimzer=Adam(learning_rate=1e-5)

    model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=[f1])

    return model
train_seq = pad_sequences(train_vectorized, maxlen=SEQ_LEN, padding='post')

val_seq = pad_sequences(val_vectorized, maxlen=SEQ_LEN, padding='post')

test_titles_seq = pad_sequences(test_titles_vectorized, maxlen=SEQ_LEN, padding='post')

test_text_seq = pad_sequences(test_text_vectorized, maxlen=SEQ_LEN, padding='post')
simple_model = build_model_simple_pavel(SEQ_LEN, emb_mat)

simple_history = simple_model.fit(

    train_seq, y_train, epochs=EPOCHS,

    batch_size=BATCH,

    validation_data=(val_seq, y_val), verbose=0

)
third_model = build_model_third_place(SEQ_LEN, emb_mat)

third_history = third_model.fit(

    train_seq, y_train, epochs=EPOCHS,

    batch_size=BATCH,

    validation_data=(val_seq, y_val), verbose=0

)
from sklearn import metrics
def evaluate(predictions, real, name=''):

    print(name)

    b_predicted = predictions > 0.5

    print("=== F1 ===")

    f1 = int(metrics.f1_score(real, b_predicted) * 100)

    print(f1)

    

    print("=== ROC-AUC ===")

    roc_auc = int(metrics.roc_auc_score(real, predictions) * 100)

    print(roc_auc)

    print()
evaluate(simple_model.predict(val_seq), y_val, 'simple')

evaluate(third_model.predict(val_seq), y_val, 'third')
print(np.mean(third_history.history['val_f1']))
plot_graphs(simple_history, "f1")

plot_graphs(simple_history, "loss")
plot_graphs(third_history, "f1")

plot_graphs(third_history, "loss")