# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os, gc, glob

"""

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

"""

# Any results you write to the current directory are saved as output.
import tensorflow as tf

from transformers import *

import tensorflow.keras.layers as L

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import KFold
train=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

sub=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
AUTO = tf.data.experimental.AUTOTUNE



tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

strategy = tf.distribute.experimental.TPUStrategy(tpu)



GCS_DS_PATH = KaggleDatasets().get_gcs_path()



print("PATH: ", GCS_DS_PATH)

print("REPLICAS: ", strategy.num_replicas_in_sync)
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
train["text"]=train["text"].apply(lambda x: str(x).strip())

test["text"]=test["text"].apply(lambda x: str(x).strip())
print("MAX_TRAIN_LEN:",train["text"].str.len().max())

print("MAX_TESTLEN:",test["text"].str.len().max())
MAX_LEN = 170
def extract_span(df):

    idx = [(full_text.index(sub_text),len(sub_text)) for full_text, sub_text in zip(df["text"].astype(str).values, df["selected_text"].astype(str).values)]

    s_idx = np.array(idx)[:, 0]

    e_idx = np.array(idx)[:, 1]+s_idx

    df["start"] = s_idx

    df["end"] = e_idx

    
extract_span(train)
def make_question(df):

    sentiments = df["sentiment"].values

    df["question"] = "What part is "+sentiments

    
make_question(train)

make_question(test)
train_ins = train[["question", "text"]].astype(str).values

test_ins = test[["question", "text"]].astype(str).values
mapped_tr = tuple(map(tuple, train_ins))

mapped_test = tuple(map(tuple, test_ins))
mapped_tr=list(mapped_tr)

mapped_test = list(mapped_test)
MODEL_TYPE = "bert-base-cased"

tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)

#base_model = TFBertModel.from_pretrained(MODEL_TYPE)
%%time

tr_tokenized = tokenizer.batch_encode_plus(mapped_tr, max_length=MAX_LEN, pad_to_max_length=True)

ts_tokenized = tokenizer.batch_encode_plus(mapped_test, max_length=MAX_LEN, pad_to_max_length=True)
tr_tokenized.keys()
train["token_ids"] = tr_tokenized["input_ids"]

train["token_type"] = tr_tokenized["token_type_ids"]

train["token_mask"] = tr_tokenized["attention_mask"]



test["token_ids"] = ts_tokenized["input_ids"]

test["token_type"] = ts_tokenized["token_type_ids"]

test["token_mask"] = ts_tokenized["attention_mask"]

"""config = base_model.config

config.output_hidden_states=True

base_model = TFBertModel(config)"""
from sklearn.model_selection import train_test_split
X_train, X_valid, = train_test_split(train.index, test_size=0.2, random_state=42)
%%time

size=10000

tr_ids = tf.convert_to_tensor(train["token_ids"][X_train])

tr_attn = tf.convert_to_tensor(train["token_mask"][X_train])

tr_id_type = tf.convert_to_tensor(train["token_type"][X_train])



tr_target_s = tf.convert_to_tensor(train["start"][X_train])

tr_target_e = tf.convert_to_tensor(train["end"][X_train])





val_ids = tf.convert_to_tensor(train["token_ids"][X_valid])

val_attn = tf.convert_to_tensor(train["token_mask"][X_valid])

val_id_type = tf.convert_to_tensor(train["token_type"][X_valid])



val_target_s = tf.convert_to_tensor(train["start"][X_valid])

val_target_e = tf.convert_to_tensor(train["end"][X_valid])





test_ids = tf.convert_to_tensor(test["token_ids"])

test_attn = tf.convert_to_tensor(test["token_mask"])

test_id_type = tf.convert_to_tensor(test["token_type"])
#BATCH_SIZE=32
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices(((tr_ids, tr_id_type, tr_attn), (tr_target_s, tr_target_e)))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices(((val_ids, val_id_type, val_attn), (val_target_s, val_target_e)))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(((test_ids, test_id_type, test_attn)))

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)
def loss_fn(y_true, y_pred):



    st_loss = tf.losses.sparse_categorical_crossentropy(y_true[0], tf.squeeze(y_pred[0]), from_logits=True)

    end_loss = tf.losses.sparse_categorical_crossentropy(y_true[1], tf.squeeze(y_pred[1]), from_logits=True)



    return st_loss+end_loss
def get_model(base_model, max_len=128):

    input_id = L.Input(shape=(max_len, ),dtype=tf.int32)

    input_type = L.Input(shape=(max_len, ),dtype=tf.int32)

    input_mask = L.Input(shape=(max_len, ),dtype=tf.int32)

    

    #ps = base_model((input_id, input_type, input_mask))[0]

    st, end = base_model((input_id, input_type, input_mask))

    

    print("Model loaded")

    model_span = tf.keras.models.Model(inputs = [input_id, input_type, input_mask], outputs = [st, end])

    

    return model_span
with strategy.scope():

    pre_model = TFBertForQuestionAnswering.from_pretrained(MODEL_TYPE)

    model_ = get_model(pre_model, MAX_LEN)

    model_.compile(loss=loss_fn, optimizer="adam")
gc.collect()
def scheduler(epoch):

  if epoch < 10:

    return 0.0001

  else:

    return 0.0001 * tf.math.exp(0.1 * (10 - epoch))



callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
hist = model_.fit(train_dataset,

           steps_per_epoch = len(X_train)//BATCH_SIZE,

           epochs=40, 

           validation_data=valid_dataset,

           validation_steps = len(X_valid)//BATCH_SIZE,

           callbacks=[callback])
gc.collect()
splits = KFold(n_splits=5, shuffle=True)
for train_idx, valid_idx in splits.split(train):

    print(train_idx, valid_idx)

    

    train_df = train[train_idx]

    valid_df = valid[valid_idx]

    

    

    
ps = model_.predict((test_ids, test_id_type, test_attn), verbose=1)
test_st = ps[0]

test_end = ps[1]
ts_s=tf.argmax(tf.nn.softmax(test_st), 1)
ts_end  = tf.argmax(tf.nn.softmax(test_end), 1)
ts_s = ts_s.numpy()

ts_end = ts_end.numpy()
#average ans span

(ts_end-ts_s).numpy().mean()  
test["selected_text"] = ""

for i in range(len(test)):

    test.loc[i, "selected_text"] = test.loc[i, "text"][ts_s[i]:ts_end[i]]
sub_df = test[["textID", "selected_text"]]
sub_df
sub_df.to_csv("submission.csv", index=False)