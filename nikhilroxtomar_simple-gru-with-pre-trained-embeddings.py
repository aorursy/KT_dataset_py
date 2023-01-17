import os

import numpy as np

import pandas as pd

import string

from tqdm import tqdm

import time

import gc



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
def clean_data(text):

    regular_punct = list(string.punctuation)

    for punc in regular_punct:

        text = text.replace(punc, f" {punc} ")

    return text



def drop_empty_rows(df):

    nan_value = float("NaN")

    df.replace("", nan_value, inplace=True)



def data_process(path):

    train_df = pd.read_csv(os.path.join(path, "train.csv"))

    test_df = pd.read_csv(os.path.join(path, "test.csv"))



    """

        Train: textID, text, selected_text, sentiment

        Test: textID, text, sentiment

    """

    ## Train

    train_df["text"] = train_df["text"].astype(str).str.lower()

    train_df["selected_text"] = train_df["selected_text"].astype(str).str.lower()

    train_df["sentiment"] = train_df["sentiment"].astype(str).str.lower()

    drop_empty_rows(train_df)



    ## Test

    test_df["text"] = test_df["text"].astype(str).str.lower()

    test_df["sentiment"] = test_df["sentiment"].astype(str).str.lower()

    drop_empty_rows(test_df)



    return train_df, test_df


import tensorflow as tf

import tensorflow.keras.backend as K



def jaccard(str1, str2):

    a = set(str1.lower().split())

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



def jaccard_distance(y_true, y_pred, smooth=100):

    y_true = tf.cast(y_true, tf.float32)

    y_pred = tf.cast(y_pred, tf.float32)

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)

    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac) * smooth
import os

import string

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Embedding, GRU, Bidirectional, Concatenate

from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, Reshape, SpatialDropout1D

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras import Model

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
def build_model(embedding_matrix, params):

    inputs = Input(shape=(params["input_size"],))

    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(inputs)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(GRU(params["gru_units"], return_sequences=True))(x)

    x = Concatenate()([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x)

    ])

    x = Dense(params["gru_units"], activation="relu")(x)

    y1 = Dense(params["input_size"], activation="softmax", name="y1")(x)

    y2 = Dense(params["input_size"], activation="softmax", name="y2")(x)

    model = Model(inputs=inputs, outputs=[y1, y2])

    return model
params = {}

params["batch_size"] = 128

params["gru_units"] = 256

params["epochs"] = 100

params["input_size"] = 32

params["embed_dim"] = 300

params["num_words"] = 70000
!ls ../input/embeddings
def load_glove(word_index):

    EMBEDDING_FILE = '../input/embeddings/glove-840B-300d.txt'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = max_features= len(word_index)+1

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

            

    return embedding_matrix 

    

def load_fasttext(word_index):    

    EMBEDDING_FILE = '../input/embeddings/wiki-news-1M-300d.vec'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = max_features= len(word_index)+1

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector



    return embedding_matrix
dataset_path = "../input/tweet-sentiment-extraction/"

train_df, test_df = data_process(dataset_path)
## Data from train

train_text = train_df["text"].apply(lambda x: clean_data(x)).values

train_selected_text = train_df["selected_text"].apply(lambda x: clean_data(x)).values

train_sentiment = train_df["sentiment"].values
## Data from test

test_text = test_df["text"].apply(lambda x: clean_data(x)).values

test_sentiment = test_df["sentiment"].values
##

train_size = len(train_text)

test_size = len(test_text)

print(f"Train data: {train_size} - Test data: {test_size}")
tokenizer = text.Tokenizer(num_words=params["num_words"], filters="")

total_text = list(train_text) + list(test_text)

tokenizer.fit_on_texts(total_text)
total_train_len = len(train_text)

x_train = []

y1_train = []

y2_train = []



for i in range(total_train_len):

    text1 = train_text[i].strip()

    text2 = train_selected_text[i].strip()



    idx1 = text1.find(text2)

    idx2 = idx1 + len(text2) - 1



    x = tokenizer.texts_to_sequences([text1])

    y1 = np.zeros((len(text1)))

    y2 = np.zeros((len(text1)))



    y1[idx1] = 1

    y2[idx2] = 1

    #y[idx1:idx2] = 1.0



    # text3 = text1[idx1:idx2]



    x = sequence.pad_sequences(x, maxlen=params["input_size"])[0]

    y1 = sequence.pad_sequences([y1], maxlen=params["input_size"])[0]

    y2 = sequence.pad_sequences([y2], maxlen=params["input_size"])[0]



    x_train.append(x)

    y1_train.append(y1)

    y2_train.append(y2)



    # print(text2)

    # print(text3)
import time

start_time = time.time()

word_index = tokenizer.word_index

embedding_matrix_1 = load_glove(word_index)

embedding_matrix_2 = load_fasttext(word_index)



total_time = (time.time() - start_time)/60.0

print("Took {0} minutes".format(total_time))
embedding_mean = np.mean([embedding_matrix_1, embedding_matrix_2], axis = 0)

del embedding_matrix_1

del embedding_matrix_2



print(np.shape(embedding_mean))
model = build_model(embedding_mean, params)

model.compile(loss=jaccard_distance, optimizer="adam")

model.summary()



x_train = np.array(x_train)

y1_train = np.array(y1_train)

y2_train = np.array(y2_train)



callbacks = [

    ReduceLROnPlateau(patience=5, monitor="val_loss", factor=0.1),

    EarlyStopping(patience=10, monitor="val_loss")

]



model.fit(

    x_train,

    [y1_train, y2_train],

    batch_size=params["batch_size"],

    epochs=params["epochs"],

    callbacks=callbacks

)
test_x = tokenizer.texts_to_sequences(test_text)

test_x = sequence.pad_sequences(test_x, maxlen=params["input_size"])

print(len(test_x))

test_y1, test_y2 = model.predict(test_x)

print(len(test_y1), len(test_y2))



answers = []

for i in range(len(test_text)):

    start = np.argmax(test_y1[i])

    end = np.argmax(test_y2[i])

    ans = test_text[i][start:end]



    regular_punct = list(string.punctuation)

    for punc in regular_punct:

        ans = ans.replace(f" {punc} ", f"{punc}")

    answers.append(ans)



submission = pd.read_csv(f"{dataset_path}/sample_submission.csv")

submission['selected_text'] = answers

submission.to_csv('submission.csv', index=False)
submission