import tensorflow as tf

import tensorflow_hub as hub

import numpy as np

import pandas as pd

from nltk import tokenize



tokenize.sent_tokenize("This is a sentence. This is another sentence.")
embed_fn = hub.load('../input/universalsentenceencoderlarge4/')
df = pd.read_csv("../input/google-quest-challenge/train.csv")

outputs = df.columns[11:]



def count_words(data):

    return len(str(data).split())



def count_words_unique(data):

    return len(np.unique(str(data).split()))



def questionowords(data):

    start_words = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'am','are','was','were','can','could','may','should','shall','does', 'do','did']

    sents = tokenize.sent_tokenize(data)

    qw = 0

    for sent in sents:

        if sent.lower().startswith(tuple(start_words)):

            qw+=1

    return qw



def questionmarks(data):

    sents = tokenize.sent_tokenize(data)

    qm = 0

    for sent in sents:

        qm += sent.count("?")  

    return qm





def get_numeric_features(df):

    df["qt_wc"] = df["question_title"].apply(count_words)

    df["qb_wc"] = df["question_body"].apply(count_words)

    df["a_wc"] = df["answer"].apply(count_words)

    df["qt_wcu"] = df["question_title"].apply(count_words_unique)

    df["qb_wcu"] = df["question_body"].apply(count_words_unique)

    df["a_wcu"] = df["answer"].apply(count_words_unique)





    df['qb_qw'] = df['question_body'].apply(questionowords)

    df['qt_qw'] = df['question_title'].apply(questionowords)

    df['qb_qm'] = df['question_body'].apply(questionmarks)

    df['qt_qm'] = df['question_title'].apply(questionmarks)

    return df



df = get_numeric_features(df)



features = ["qt_wc", "qb_wc", "a_wc", "qt_wcu", "qb_wcu", "a_wcu",

            "qb_qw", "qt_qw", "qb_qm", "qt_qm"]

from tqdm import tqdm_notebook



MAX_SEQ = 30



def get_sentences(x):

    sentences = [s for s in tokenize.sent_tokenize(x) if s != ""]

    if len(sentences) > MAX_SEQ:

        return sentences[:MAX_SEQ]

    return sentences + [""]*(MAX_SEQ - len(sentences))





def get_use(df):

    QT = embed_fn(df["question_title"].values)["outputs"].numpy()



    A = np.zeros((df.shape[0], MAX_SEQ, 512), dtype=np.float32)

    for i, x in tqdm_notebook(list(enumerate(df["answer"].values))):

        A[i] = embed_fn(get_sentences(x))["outputs"].numpy()



    QB = np.zeros((df.shape[0], MAX_SEQ, 512), dtype=np.float32)

    for i, x in tqdm_notebook(list(enumerate(df["question_body"].values))):

        QB[i] = embed_fn(get_sentences(x))["outputs"].numpy()



    return QT, A, QB



QT, A, QB = get_use(df)
import tensorflow.keras.layers as KL





def nn_block(input_layer, size, dropout_rate, activation):

    out_layer = KL.Dense(size, activation=None)(input_layer)

    #out_layer = KL.BatchNormalization()(out_layer)

    out_layer = KL.Activation(activation)(out_layer)

    out_layer = KL.Dropout(dropout_rate)(out_layer)

    return out_layer



def cnn_block(input_layer, size, dropout_rate, activation):

    out_layer = KL.Conv1D(size, 1, activation=None)(input_layer)

    #out_layer = KL.LayerNormalization()(out_layer)

    out_layer = KL.Activation(activation)(out_layer)

    out_layer = KL.Dropout(dropout_rate)(out_layer)

    return out_layer

    

def get_model():

    qt_input = KL.Input(shape=(QT.shape[1],))



    a_input = KL.Input(shape=(A.shape[1], A.shape[2]))

    qb_input = KL.Input(shape=(QB.shape[1], QB.shape[2]))



    dummy_input = KL.Input(shape=(1,))



    a_emb = KL.Flatten()(KL.Embedding(2, 8)(dummy_input))

    qb_emb = KL.Flatten()(KL.Embedding(2, 8)(dummy_input))



    embs = KL.concatenate([KL.RepeatVector(MAX_SEQ)(a_emb), KL.RepeatVector(MAX_SEQ)(qb_emb)], axis=-2)



    x = KL.concatenate([KL.SpatialDropout1D(0.7)(KL.RepeatVector(2*MAX_SEQ)(qt_input)), 

                        KL.SpatialDropout1D(0.3)(KL.concatenate([a_input, qb_input], axis=-2))])

    x = KL.concatenate([x, embs])



    x = cnn_block(x, 256, 0.1, "relu")

    x = KL.concatenate([KL.GlobalAvgPool1D()(x), KL.GlobalMaxPool1D()(x)])



    feature_input = KL.Input(shape=(len(features),))



    hidden_layer = KL.concatenate([KL.BatchNormalization()(feature_input), x])

    hidden_layer = nn_block(hidden_layer, 128, 0.1, "relu")



    out = KL.Dense(len(outputs), activation="sigmoid")(hidden_layer)



    model = tf.keras.models.Model(inputs=[qt_input, a_input, qb_input, feature_input, dummy_input], outputs=out)

    return model
from sklearn.model_selection import KFold, GroupKFold

from sklearn.metrics import mean_squared_error

from tensorflow.keras.optimizers import Nadam

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.backend import epsilon

import tensorflow.keras.backend as K

import os



NUM_FOLDS = 10

BATCH_SIZE = 32

MODEL_FOLDER = "use_models/"

os.mkdir(MODEL_FOLDER)



models_w = []



y = df[outputs].copy()

for col in outputs:

    y[col] = y[col].rank(method="average")

y = MinMaxScaler().fit_transform(y.values)



y_oof = np.zeros(y.shape)



kfold = GroupKFold(NUM_FOLDS)

for fold, (train_ind, val_ind) in enumerate(kfold.split(y, y, groups=df["question_body"].values)):

    model_path = "{folder}model{fold}.h5".format(folder=MODEL_FOLDER, fold=fold)

    print(model_path)

    train_df, val_df = df.iloc[train_ind].copy(), df.iloc[val_ind].copy()

    y_train, y_val = y[train_ind], y[val_ind]



    model = get_model()

    for lr, epochs in [(0.0002, 4), (0.002, 4), (0.0005, 3), (0.0002, 3)]: 

        model.compile(loss="binary_crossentropy", optimizer=Nadam(lr=lr))

        hist = model.fit([QT[train_ind], A[train_ind], QB[train_ind], train_df[features].values, np.ones(train_df.shape[0])], y_train,

                         batch_size=BATCH_SIZE, epochs=epochs,

                         validation_data=([QT[val_ind], A[val_ind], QB[val_ind], val_df[features].values, np.ones(val_df.shape[0])], y_val),

                         verbose=0, shuffle=True)



    y_oof[val_ind, :] = model.predict([QT[val_ind], A[val_ind], QB[val_ind], val_df[features].values, np.ones(val_df.shape[0])], batch_size=BATCH_SIZE, verbose=0)

    model.save_weights(model_path)

    K.clear_session()

from scipy.stats import spearmanr



def evaluate(y, y_pred, verbose=False):

    score = 0

    for i in range(y.shape[1]):

        col_score = spearmanr(y[:, i], y_pred[:, i])[0]

        if verbose:

            print(outputs[i], np.round(col_score, 3))

        score += col_score/y.shape[1]

    return np.round(score, 3)



evaluate(y, y_oof, True)
use_df = pd.DataFrame(y_oof, columns=outputs)

use_df["qa_id"] = df["qa_id"].values



use_df.to_csv("use_oof.csv", index=False)

use_df.head()