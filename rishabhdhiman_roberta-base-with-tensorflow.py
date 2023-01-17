import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from wordcloud import WordCloud, STOPWORDS
import tensorflow as tf
import missingno as msno
from collections import defaultdict
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers
import tensorflow.keras.backend as K

%matplotlib inline
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv").fillna("")
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
sample_submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
train["text"] = train["text"].apply(lambda x : x.strip())
test["text"] = test["text"].apply(lambda x : x.strip())
tf.__version__
MAX_LEN = 128
PATH = "../input/tf-roberta/"
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file = PATH + "vocab-roberta-base.json",
    merges_file = PATH + "merges-roberta-base.txt",
    lowercase = True,
    add_prefix_space = True
)

sentiment_id = {"positive": 1313, "negative": 2430, "neutral": 7974}

# Make tokens
# Make input ids  0-start, 2-end, 1-pad
# make attention masks
# make start_token
# make end token
num_rows = train.shape[0]
train_input_ids = []
train_attention_masks = []
train_start_tokens = []
train_end_tokens = []

for row in range(num_rows):
    encoding = tokenizer.encode(train.iloc[row, 1])
    text_tokens = encoding.tokens
    senti = tokenizer.encode(train.iloc[row, 3])
    padding = MAX_LEN - len(encoding.ids)
    input_id = [0] + encoding.ids + [2, 2] + senti.ids + [2] + [1] * (padding - 5)
    attention_mask = [1] * len([0] + encoding.ids + [2, 2] + senti.ids + [2]) + [0] * (padding - 5)
    selected_tokens = tokenizer.encode(train.iloc[row, 2]).tokens
    tok_start = 0
    for tok_s in selected_tokens:
        for i, tok_t in enumerate(text_tokens):
            if tok_t == tok_s:
                tok_start = i + 1
                break  
        break
    tok_end = tok_start + len(selected_tokens) - 1
    start_tok, end_tok = [0] * len(input_id), [0] * len(input_id)
    start_tok[tok_start] = 1
    end_tok[tok_end] = 1
    
    train_input_ids.append(input_id)
    train_attention_masks.append(attention_mask)
    train_start_tokens.append(start_tok)
    train_end_tokens.append(end_tok)
    
train_input_ids = np.array(train_input_ids, dtype = "int32")
train_attention_masks = np.array(train_attention_masks, dtype = "int32")
train_start_tokens = np.array(train_start_tokens, dtype = "int32")
train_end_tokens = np.array(train_end_tokens, dtype = "int32")
train_token_type_ids = np.zeros((num_rows,MAX_LEN), dtype = "int32")




num_rows = test.shape[0]
test_input_ids = []
test_attention_masks = []

for row in range(num_rows):
    encoding = tokenizer.encode(test.iloc[row, 1])
    text_tokens = encoding.tokens
    senti = tokenizer.encode(test.iloc[row, -1])
    padding = MAX_LEN - len(encoding.ids)
    input_id = [0] + encoding.ids + [2, 2] + senti.ids + [2] + [1] * (padding - 5)
    attention_mask = [1] * len([0] + encoding.ids + [2, 2] + senti.ids + [2]) + [0] * (padding - 5)
    test_input_ids.append(input_id)
    test_attention_masks.append(attention_mask)
    
test_input_ids = np.array(test_input_ids, dtype = "int32")
test_attention_mask = np.array(test_attention_masks, dtype = "int32")    
test_token_type_ids = np.zeros((num_rows,MAX_LEN), dtype = "int32")
ids = tf.keras.layers.Input((MAX_LEN, ), dtype = tf.int32)
att = tf.keras.layers.Input((MAX_LEN, ), dtype = tf.int32)
token = tf.keras.layers.Input((MAX_LEN, ), dtype = tf.int32)

config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)

x = bert_model(ids,attention_mask=att,token_type_ids=token)
# training for starting indices
x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
x1 = tf.keras.layers.Conv1D(128, 1,padding='same')(x1)
x1 = tf.keras.layers.Conv1D(1,1)(x1)
x1 = tf.keras.layers.Flatten()(x1)
x1 = tf.keras.layers.Activation('softmax')(x1)
# training for end indices
x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
x2 = tf.keras.layers.Conv1D(128, 1,padding='same')(x2)
x2 = tf.keras.layers.Conv1D(1,1)(x2)
x2 = tf.keras.layers.Flatten()(x2)
x2 = tf.keras.layers.Activation('softmax')(x2)

model = tf.keras.models.Model(inputs=[ids, att, token], outputs=[x1,x2])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()
model.fit([train_input_ids.reshape(train.shape[0], 128), train_attention_masks.reshape(train.shape[0], 128), train_token_type_ids.reshape(train.shape[0], 128)], 
          [train_start_tokens.reshape(train.shape[0], 128), train_end_tokens.reshape(train.shape[0], 128)], epochs=3, batch_size=32) 
preds = model.predict([test_input_ids.reshape(test.shape[0], 128), test_attention_mask.reshape(test.shape[0], 128), test_token_type_ids.reshape(test.shape[0], 128)])
preds = []
for k in range(test_input_ids.shape[0]):
    a = np.argmax(preds[0][k])
    b = np.argmax(preds[1][k])
    if a>b: 
        st = test.iloc[k,1]
    else:
        st = tokenizer.decode(test_input_ids[k][a:b + 1])
    preds.append(st)