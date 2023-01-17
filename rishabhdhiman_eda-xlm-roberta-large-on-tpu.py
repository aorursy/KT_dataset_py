import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

import string

import re

import random

import missingno as msno

import tensorflow as tf

from transformers import TFAutoModel, AutoTokenizer

import os

from tqdm.notebook import tqdm

import tensorflow_hub as hub

from sklearn.model_selection import StratifiedKFold

import tensorflow.keras.backend as K

%matplotlib inline
train1 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train2 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")



train2.toxic = train2.toxic.round().astype(int)



valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv') 
train = pd.concat([

    train1[['comment_text', 'toxic']],

    train2[['comment_text', 'toxic']]

])
plt.figure(figsize = (15, 8))

plt.title("Count of labels 0 vs 1")

plt.xlabel("Toxic")

plt.ylabel("Count")

sns.countplot(x = "toxic", data = train)
train = pd.concat([

    train1[['comment_text', 'toxic']],

    train2[['comment_text', 'toxic']].query('toxic==1')

#     train2[['comment_text', 'toxic']].query('toxic==0').sample(n=30000, random_state=0)

]).sample(100000, random_state = 1)

train = pd.concat([

    train,

    valid[["comment_text", "toxic"]]

])
plt.figure(figsize = (15, 8))

plt.title("Count of labels 0 vs 1")

plt.xlabel("Toxic")

plt.ylabel("Count")

sns.countplot(x = "toxic", data = train)
plt.figure(figsize = (12, 8))

msno.bar(train)
stopwords = set(STOPWORDS)



def  word_cloud(data, title =None):

    data = data.apply(lambda x : x.lower())

    cloud = WordCloud(

    background_color = "black",

    stopwords = stopwords,

    max_words = 200,

    max_font_size = 40,

    scale = 3).generate(str(data))

    

    fig = plt.figure(figsize= (15, 15))

    plt.axis("off")

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.25)



    plt.imshow(cloud)

    plt.show()
word_cloud(train["comment_text"], "WordCloud for train data")
word_cloud(valid["comment_text"].apply(str), "WordCloud for valid data")
word_cloud(test["content"], "WordCloud for test data")
plt.figure(figsize = (15, 8))

len_sent = train["comment_text"].apply(lambda x : len(x.split()))

sns.distplot(len_sent.values)

plt.title("Distribution of length of words")

plt.xlabel("Length of words")

plt.ylabel("Probability of occurance")
print(f"Max length of characters = {len_sent.values.max()}")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

tokenizer = AutoTokenizer.from_pretrained("jplu/tf-xlm-roberta-large")
exp_text = "I am currently participating in Jigsaw competition"

tokenizer.tokenize(exp_text)
MAX_LEN = 192
def preprocess(data, max_seq_length = MAX_LEN, tokenizer = tokenizer):    

    ids = []

    masks = []

    segment = []

    for i in tqdm(range(len(data))):

        

        tokens = tokenizer.tokenize(data[i])

        if len(tokens) > max_seq_length - 2:

            tokens = tokens[ : max_seq_length - 2]



        # Converting tokens to ids

        input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])



        # Input mask

        input_masks = [1] * len(input_ids)



        # padding upto max length

        padding = max_seq_length - len(input_ids)

        input_ids.extend([0] * padding)

        input_masks.extend([0] * padding)

        segment_ids =[0]* max_seq_length

        

        

        ids.append(input_ids)

        masks.append(input_masks)

        segment.append(segment_ids)

    

    return (np.array(ids), np.array(masks), np.array(segment))

train_ids, train_masks, train_segment =  preprocess(train["comment_text"].values)
test_ids, test_masks, test_segment =  preprocess(test["content"].values)
valid_ids, valid_masks, valid_segment =  preprocess(valid["comment_text"].values)
y_train = train["toxic"].values

y_valid = valid["toxic"].values
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
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
def model(roberta_layer, max_len = MAX_LEN):

    

        input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

        input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

        segment_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



        pooled_output, sequence_output = roberta_layer([input_word_ids, input_mask, segment_ids])



        # There are two outputs: a pooled_output of shape [batch_size, 768] with representations for 

        # the entire input sequences and a sequence_output of shape [batch_size, max_seq_length, 768] 

        # with representations for each input token (in context)





        x = pooled_output

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(128, activation = "relu")(x)

        x = tf.keras.layers.Dense(1, activation = "sigmoid")(x)



        model = tf.keras.Model(inputs = [input_word_ids, input_mask, segment_ids], outputs = x)

        return model
with strategy.scope():

    roberta_layer = TFAutoModel.from_pretrained("jplu/tf-xlm-roberta-large", trainable = True)

    model = model(roberta_layer)

    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss="binary_crossentropy", metrics=["accuracy"])

model.summary()
skf = StratifiedKFold(n_splits=5, shuffle = True)

skf.get_n_splits(train_ids, y_train)

skf
i = 1

preds = []

for train_index, test_index in skf.split(train_ids, y_train):

    print("\n")

    print("#" * 20)

    print(f"FOLD No {i}")

    print("#" * 20)

    

    

    tr_ids = train_ids[train_index]

    tr_masks = train_masks[train_index]

    tr_segment = train_segment[train_index]

    

    vd_ids = train_ids[test_index]

    vd_masks = train_masks[test_index]

    vd_segment = train_segment[test_index]

    

    y_tr = y_train[train_index]

    y_vd = y_train[test_index]

    

    

    history = model.fit(

    (tr_ids, tr_masks, tr_segment), y_tr,

    epochs=2,

    batch_size=BATCH_SIZE,

    validation_data = ((vd_ids, vd_masks, vd_segment), y_vd),

    steps_per_epoch = len(tr_ids)//BATCH_SIZE)



    predictions = model.predict((test_ids, test_masks, test_segment))

    preds.append(predictions)

    

    i += 1

    K.clear_session()
predictions =  (preds[0] + preds[1] + preds[2] + preds[3] + preds[4])/5
# predictions = model.predict((test_ids, test_masks, test_segment))

sub["toxic"] = predictions

sub.set_index("id", inplace = True)

sub.to_csv("submission.csv")