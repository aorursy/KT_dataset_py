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
BATCH_SIZE = 32 
#train["text"]=train["text"].apply(lambda x: str(x).strip())

test["text"]=test["text"].apply(lambda x: str(x).strip())
#print("MAX_TRAIN_LEN:",train["text"].str.len().max())

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

    
#make_question(train)

make_question(test)
#train_ins = train[["question", "text"]].astype(str).values

test_ins = test[["question", "text"]].astype(str).values
#mapped_tr = tuple(map(tuple, train_ins))

mapped_test = tuple(map(tuple, test_ins))
#mapped_tr=list(mapped_tr)

mapped_test = list(mapped_test)
!ls /kaggle/input
MODEL_DIR =  "/kaggle/input/bertlatgetweetqa/"
MODEL_TYPE = "bert-base-cased"

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR+"tokenizer")

#base_model = TFBertModel.from_pretrained(MODEL_TYPE)
%%time

#tr_tokenized = tokenizer.batch_encode_plus(mapped_tr, max_length=MAX_LEN, pad_to_max_length=True)

ts_tokenized = tokenizer.batch_encode_plus(mapped_test, max_length=MAX_LEN, pad_to_max_length=True)
#tr_tokenized.keys()
#train["token_ids"] = tr_tokenized["input_ids"]

#train["token_type"] = tr_tokenized["token_type_ids"]

#train["token_mask"] = tr_tokenized["attention_mask"]



test["token_ids"] = ts_tokenized["input_ids"]

test["token_type"] = ts_tokenized["token_type_ids"]

test["token_mask"] = ts_tokenized["attention_mask"]

"""config = base_model.config

config.output_hidden_states=True

base_model = TFBertModel(config)"""
from sklearn.model_selection import train_test_split
%%time

size=10000



test_ids = tf.convert_to_tensor(test["token_ids"])

test_attn = tf.convert_to_tensor(test["token_mask"])

test_id_type = tf.convert_to_tensor(test["token_type"])
#BATCH_SIZE=32


test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(((test_ids, test_id_type, test_attn)))

    .batch(BATCH_SIZE)

    #.prefetch(AUTO)

)
def loss_fn(y_true, y_pred):



    st_loss = tf.losses.sparse_categorical_crossentropy(y_true[0], tf.squeeze(y_pred[0]), from_logits=True)

    end_loss = tf.losses.sparse_categorical_crossentropy(y_true[1], tf.squeeze(y_pred[1]), from_logits=True)



    return st_loss+end_loss
pre_model = TFBertForQuestionAnswering.from_pretrained(MODEL_DIR+"model/")

model_ = pre_model

model_.compile(loss=loss_fn, optimizer="adam")
gc.collect()
ps = model_.predict((test_ids, test_id_type, test_attn), verbose=1)
test_st = ps[0]

test_end = ps[1]
ts_s=tf.argmax(tf.nn.softmax(test_st), 1)

ts_end  = tf.argmax(tf.nn.softmax(test_end), 1)
ts_s = ts_s.numpy()

ts_end = ts_end.numpy()
#average ans span

(ts_end-ts_s).mean()  
ts_end<ts_s
test["selected_text"] = ""

for i in range(len(test)):

    if(ts_s[i]<ts_end[i]):

        test.loc[i, "selected_text"] = test.loc[i, "text"][ts_s[i]:ts_end[i]]

    else:

        test.loc[i, "selected_text"] = test.loc[i, "text"]
sub_df = test[["textID", "selected_text"]]
sub_df
sub_df.to_csv("submission.csv", index=False)