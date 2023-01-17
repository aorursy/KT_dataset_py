"""

IMPORT

"""



import math, re, os

import tensorflow as tf

from tqdm import tqdm

import numpy as np

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import io

import json





import numpy as np

import pandas as pd

from tensorflow.keras.models import Model

from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from tensorflow.keras.preprocessing import text, sequence

from gensim.models import KeyedVectors



from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential



import transformers

from tokenizers import BertWordPieceTokenizer
# Adapted from https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu



PATH_TPU_WORKER = ''



def check_tpu():

    """

    Detect TPU hardware and return the appopriate distribution strategy

    """

    

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 

        print('Running on TPU: {}'.format(tpu.master()))

    except ValueError:

        tpu = None



    if tpu:

        tf.config.experimental_connect_to_cluster(tpu)

        tf.tpu.experimental.initialize_tpu_system(tpu)

        tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    else:

        tpu_strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



    print("Num. replicas: {}".format(tpu_strategy.num_replicas_in_sync))

    

    return tpu, tpu_strategy

    

tpu, tpu_strategy = check_tpu()

PATH_TPU_WORKER = tpu.master()

NUM_REPLICAS = tpu_strategy.num_replicas_in_sync
%%time



"""

PATH

"""



PATH_CHALLENGE = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/'



PATH_TRAIN_FILENAME = PATH_CHALLENGE + "jigsaw-toxic-comment-train.csv"

PATH_TEST_FILENAME = PATH_CHALLENGE + "test.csv"

PATH_VALID_FILENAME = PATH_CHALLENGE + "validation.csv"





"""

LOAD

"""



train_df = pd.read_csv(PATH_TRAIN_FILENAME)

test_df = pd.read_csv(PATH_TEST_FILENAME)

valid_df = pd.read_csv(PATH_VALID_FILENAME)



"""

PREPROCESSING

"""



MAX_LEN = 256



# Adapted https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')



save_path = '/kaggle/working/distilbert_base_uncased/'

if not os.path.exists(save_path):

    os.makedirs(save_path)

tokenizer.save_pretrained(save_path)



fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt', lowercase=True)





def encode(texts, tokenizer, chunk_size=256, maxlen=MAX_LEN):

    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

    all_ids = []

    

    for i in range(0, len(texts), chunk_size):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)



x_train = encode(train_df.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_valid = encode(valid_df.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_test = encode(test_df.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)



y_train = train_df.toxic.values

y_valid = valid_df.toxic.values
"""

SETTINGS

"""



AUTO = tf.data.experimental.AUTOTUNE



BATCH_SIZE = 16 

TOTAL_BATCH_SIZE = BATCH_SIZE * tpu_strategy.num_replicas_in_sync

print("Batch size: {}".format(BATCH_SIZE))

print("Total batch size: {}".format(TOTAL_BATCH_SIZE))





"""

DATA Loading

"""





train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(TOTAL_BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    #.repeat()

    .batch(TOTAL_BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(TOTAL_BATCH_SIZE)

    #.repeat()

    .cache()

    .prefetch(AUTO)

)
def simple_model(max_len=MAX_LEN):

    words = Input(shape=(max_len,), batch_size=TOTAL_BATCH_SIZE, dtype=tf.int32, name="words")

    x = Dense(10, activation='relu')(words)

    out = Dense(1, activation='sigmoid')(x)

    

    model = Model(inputs=words, outputs=out)

    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model.summary()



simple_model()
with tpu_strategy.scope():

    simple_model()
"""

DEFINE MODEL

"""





def lstm_model(vocab_size, max_len=MAX_LEN):

    

    words = Input(shape=(max_len,), dtype=tf.int32, name="words")

    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=512, input_length=max_len)(words)

    x = tf.keras.layers.SpatialDropout1D(0.3)(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)



    hidden = tf.keras.layers.concatenate([

        tf.keras.layers.GlobalMaxPooling1D()(x),

        tf.keras.layers.GlobalAveragePooling1D()(x),

    ])

    hidden = tf.keras.layers.add([hidden, Dense(4 * 256, activation='relu')(hidden)])

    hidden = tf.keras.layers.add([hidden, Dense(4 * 256, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    

    model = tf.keras.Model(inputs=words, outputs=result)



    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), # much more faster with 

                  optimizer=tf.keras.optimizers.Adam(1e-4),

                  metrics=['accuracy'])

    return model



"""

BUILD

"""



with tpu_strategy.scope():

    vocab_size = tokenizer.vocab_size # Distil

    model = lstm_model(vocab_size)

model.summary()
%%time



"""

TRAIN

"""



EPOCHS = 5



N_TRAIN_STEPS = 219

N_VALID_STEPS = 63

train_history = model.fit(

    train_dataset,

    steps_per_epoch=N_TRAIN_STEPS,

    validation_data=valid_dataset,

    validation_steps=N_VALID_STEPS,

    epochs=EPOCHS

)





def auc_roc(dataset, ground_truth):

    from sklearn.metrics import roc_curve

    y_pred_keras = model.predict(dataset, verbose=1).ravel()

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(ground_truth, y_pred_keras)

    from sklearn.metrics import auc

    return auc(fpr_keras, tpr_keras)



print("AUC-ROC validation set: ")

auc_roc(valid_dataset, y_valid)
predictions = model.predict(test_dataset, verbose=1).ravel()



input_size = test_df.shape[0]

output_size = predictions.shape[0]



if input_size != output_size:

    print("Input size differs from output size. Input size: {}, Output size: {}".format(input_size,output_size))



if output_size % NUM_REPLICAS == 0:

    print("Predicitions is divisible by ".format(NUM_REPLICAS))

    

    

submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    'toxic': predictions[:input_size]

})



print("Save submission to csv.")

submission.to_csv('submission.csv', index=False)