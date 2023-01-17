# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train.head()
train["toxic"].value_counts()
train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
train2.head()
t1 = train[train["toxic"]==0]

t2 = train[train["toxic"]==1]

t1 = t1.sample(frac=1).reset_index(drop=True)

t1 = t1.iloc[:60000,:]

train = pd.DataFrame()

train=train.append(t1)

train = train.append(t2)
train["toxic"].value_counts()
def change(num):

    if(num==0):

        return 0

    elif(num>=.6):

        return 1

    else:

        return 2
train2["toxic"] = train2["toxic"].apply(change)
train21 = train2[train2["toxic"]==1]

train20 = train2[train2["toxic"]==0]
train20 = train20.iloc[:100000,:]
train2 = pd.DataFrame()

train2 = train2.append(train20)

train2 = train2.append(train21)
train = train.iloc[:,1:3]
train2 = train2.iloc[:,1:3]
train = train.append(train2)
train["toxic"].value_counts()
train = train.sample(frac=1).reset_index(drop=True)

validation = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv")
validation.head()
test = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")
test.head()
!pip install transformers
import os



import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import transformers

from transformers import TFAutoModel, AutoTokenizer

from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

from sklearn.metrics import roc_auc_score

from tensorflow.keras.callbacks import ModelCheckpoint

import traitlets

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import tqdm
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):

    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)

def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])
def build_model(transformer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

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





# Configuration

EPOCHS = 2

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 192

MODEL = 'jplu/tf-xlm-roberta-large'

tokenizer = AutoTokenizer.from_pretrained(MODEL)
# # First load the real tokenizer

# tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# # Save the loaded tokenizer locally

# tokenizer.save_pretrained('.')

# # Reload it with the huggingface tokenizers library

# fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)

# fast_tokenizer
# x_train = fast_encode(train["comment_text"].astype(str), fast_tokenizer, maxlen=MAX_LEN)

# x_valid = fast_encode(validation["comment_text"].astype(str), fast_tokenizer, maxlen=MAX_LEN)

# x_test = fast_encode(test["content"].astype(str), fast_tokenizer, maxlen=MAX_LEN)



x_train = regular_encode(train["comment_text"].astype(str), tokenizer, maxlen=MAX_LEN)

x_valid = regular_encode(validation["comment_text"].astype(str), tokenizer, maxlen=MAX_LEN)

x_test = regular_encode(test["content"].astype(str), tokenizer, maxlen=MAX_LEN)



y_train = train.toxic.values

y_valid = validation.toxic.values
# del(train)

# del(test)

# del(validation)
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
%%time

with strategy.scope():

    transformer_layer = TFAutoModel.from_pretrained(MODEL)

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
n_steps = x_train.shape[0] // BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS

)
n_steps = x_valid.shape[0] // BATCH_SIZE

train_history_2 = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    epochs=EPOCHS*2

)
value = model.predict(test_dataset, verbose=1)

# value = pd.DataFrame(value)

# value.to_csv('submission.csv', index=False)
sub = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
ll = []

value = list(value)

for i in value:

    a = i[0]

    if(a>=0.5):

        ll.append(1)

    else:

        ll.append(0)
len(ll)
ll[:20]
sub["toxic"] = ll
sub.head()
sub.to_csv("submission.csv",index = False)