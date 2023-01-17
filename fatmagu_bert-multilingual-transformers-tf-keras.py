# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets

import transformers

from tqdm.notebook import tqdm

from tokenizers import BertWordPieceTokenizer

from keras.callbacks import Callback

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras import backend as K



def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
class RocAucCallback(Callback):

    def __init__(self, test_data, score_thr):

        self.test_data = test_data

        self.score_thr = score_thr

        self.test_pred = []

        

    def on_epoch_end(self, epoch, logs=None):

        if logs['val_auc_5'] > self.score_thr:

            print('\nRun TTA...')

            for td in self.test_data:

                self.test_pred.append(self.model.predict(td))
def fast_encode (texts, tokenizer, chunk_size=256, maxlen=512):

    tokenizer.enable_truncation(max_length = maxlen)

    tokenizer.enable_padding(max_length = maxlen)

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].to_list()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

     

    return np.array(all_ids)
def build_model (transformer, max_len=512, loss='binary_crossentropy', lr=1e-5):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

    

    model = Model(input_word_ids,out)

    model.compile(Adam(lr = lr), loss = loss, metrics = ['accuracy',tf.keras.metrics.AUC(),f1_m, precision_m, recall_m])

    

    return model
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')



# Configuration

EPOCHS = 10

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 512
GCS_DS_PATH
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

tokenizer.save_pretrained('.')

fast_tokenizer = BertWordPieceTokenizer("vocab.txt", lowercase=False)

train_toxic_data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train_unintended_bias_data =  pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
train_unintended_bias_data.toxic = train_unintended_bias_data.toxic.round().astype(int)

train_data = pd.concat([train_toxic_data[["comment_text", "toxic"]], train_unintended_bias_data[["comment_text", "toxic"]]])
print(len(train_data))
train_data_sample = train_data.sample(100000)
validation_data =  pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv")

test_data =  pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")

sub_data =  pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
y_train = train_data_sample.toxic.values

y_valid = validation_data.toxic.values
x_train = fast_encode(train_data_sample.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_valid = fast_encode(validation_data.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

x_test = fast_encode(test_data.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)
traind_dataset = (

    tf.data.Dataset.from_tensor_slices((x_train,y_train)).repeat().shuffle(2000).batch(BATCH_SIZE).prefetch(AUTO))



valid_dataset = (

    tf.data.Dataset.from_tensor_slices((x_valid,y_valid)).batch(BATCH_SIZE).cache().prefetch(AUTO))



test_dataset = (

    tf.data.Dataset.from_tensor_slices((x_test)).batch(BATCH_SIZE))

with strategy.scope():

    transformer_layer = (transformers.TFBertModel.from_pretrained('bert-base-multilingual-cased'))

    model = build_model(transformer_layer, lr=1e-3, max_len=MAX_LEN)

model.summary()
#metric_callbacks = RocAucCallback(test_dataset, 0.91)

n_steps = x_train.shape[0] // BATCH_SIZE

train_history = model.fit(traind_dataset, 

                         steps_per_epoch=n_steps,

                         validation_data=valid_dataset,

                         epochs=EPOCHS)
sub_data['toxic'] = model.predict(test_dataset, verbose=1)

sub_data.to_csv("submission.csv", index=False)