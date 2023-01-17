import warnings

warnings.filterwarnings('ignore')



import os

import time

import gc

import sys

import torch

import re

import math

import itertools



import numpy as np

import pandas as pd

import random

import shutil

import pickle

from functools import partial

from joblib import Parallel, delayed

from operator import itemgetter

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score



import tensorflow as tf

import tensorflow.keras.backend as K

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

from transformers import PreTrainedModel

import transformers

import tokenizers



from tqdm._tqdm_notebook import tqdm_notebook as tqdm

from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import StratifiedKFold

from transformers.modeling_tf_utils import get_initializer

from nltk import sent_tokenize



tqdm.pandas()

print(os.listdir("../input/"))
def seed_everything(seed=88888):

    

    # Python/TF Seeds

    random.seed(seed)

    np.random.seed(seed)

    os.environ['XLA_USE_BF16'] = '1'

    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)



    # Torch Seeds

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False



seed_everything(88888)
start_time = time.time()

print("Loading Model Parameters ...")



# Model Version

model_version = "1.0.0"



# Model Parameters

n_epochs = 1

n_splits = 3

max_length = 96

train_batch_size = 32

model_seed = 88888



# Model Directories

directory = "../input/"

roberta_directory = directory + "tf-roberta/"



# Model File Paths

train_file = directory + 'tweet-sentiment-extraction/train.csv'

complete_file = directory + 'complete-tweet-sentiment-extraction-data/tweet_dataset.csv'

test_file = directory + 'tweet-sentiment-extraction/test.csv'

sample_file = directory + 'tweet-sentiment-extraction/sample_submission.csv'



# Model Tokenizer

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file = roberta_directory + 'vocab-roberta-base.json', 

    merges_file = roberta_directory + 'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)

print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

print("Loading Initial Data ...")



df_train = pd.read_csv(train_file)

df_complete = pd.read_csv(complete_file)



df_train['text'] = df_train['text'].astype(str).str.strip()

df_train['selected_text'] = df_train['selected_text'].astype(str)

df_train.fillna('', inplace=True, axis=1)



df_complete = df_complete[["aux_id", "text", "sentiment"]]

df_complete.columns = ["textID", "text", "sentiment_main"] 

df_train = pd.merge(df_train, df_complete[["textID", "sentiment_main"]], on="textID", how="left")

df_train = df_train.dropna(axis=0).reset_index(drop=True)

df_train = df_train[["textID", "text", "sentiment_main", "sentiment"]]



df_secondary = df_complete[~df_complete["textID"].isin(df_train["textID"].values)]

df_secondary = df_secondary.dropna(axis=0).reset_index(drop=True)



print("--- %s seconds ---" % (time.time() - start_time))
df_train
df_secondary
start_time = time.time()

print("Preprocess Training Data ...")



ct = df_train.shape[0]

target_mapping = {"positive": 0, "negative": 1, "neutral": 2}



input_ids = np.ones((ct, max_length), dtype='int32')

attention_mask = np.zeros((ct, max_length), dtype='int32')

token_type_ids = np.zeros((ct, max_length), dtype='int32')



for k in tqdm(range(df_train.shape[0]), total=df_train.shape[0]):



    # Establish Initial Preprocessing

    text1 = " " + " ".join(df_train.loc[k, 'text'].split())

    s_tok = tokenizer.encode(df_train.loc[k,'sentiment_main']).ids[0]

    encoded_ids = [0, s_tok] + tokenizer.encode(text1).ids[0:max_length - 3]

    input_ids[k, :len(encoded_ids) + 3] = [0, s_tok] + encoded_ids + [2]

    attention_mask[k, :len(encoded_ids) + 3] = 1

    

print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

print("Preprocess Labels ...")



encoder = OneHotEncoder(sparse=False)

target_values = np.array(df_train["sentiment"].map(target_mapping).values)

target_values = target_values.reshape(len(target_values), 1)

target_values = encoder.fit_transform(target_values)



print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

print("Preprocess Secondary Data ...")



ct = df_secondary.shape[0]

input_ids_2 = np.ones((ct, max_length), dtype='int32')

attention_mask_2 = np.zeros((ct, max_length), dtype='int32')

token_type_ids_2 = np.zeros((ct, max_length), dtype='int32')



for k in tqdm(range(df_secondary.shape[0]), total=df_secondary.shape[0]):



    # Establish Initial Preprocessing

    text1 = " " + " ".join(df_train.loc[k, 'text'].split())

    s_tok = tokenizer.encode(df_train.loc[k,'sentiment_main']).ids[0]

    encoded_ids = [0, s_tok] + tokenizer.encode(text1).ids[0:max_length - 3]

    input_ids_2[k, :len(encoded_ids) + 3] = [0, s_tok] + encoded_ids + [2]

    attention_mask_2[k, :len(encoded_ids) + 3] = 1



print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

print("Loading Model Pipeline ...")



def loss_fn(y_true, y_pred):

    loss = tf.nn.softmax_cross_entropy_with_logits(

        y_true, y_pred, axis=-1, name=None

    )

    loss = tf.math.reduce_mean(loss)

    return loss



def build_model():



    # Establish Inputs

    input_word_ids = tf.keras.layers.Input((max_length,), dtype=tf.int32)

    attention_mask = tf.keras.layers.Input((max_length,), dtype=tf.int32)

    token_type_ids = tf.keras.layers.Input((max_length,), dtype=tf.int32)

    padding = tf.cast(tf.equal(input_word_ids, 1), tf.int32)



    # Pad Inputs by Batch Max Length

    length = max_length - tf.reduce_sum(padding, -1)

    max_batch_length = tf.reduce_max(length)

    input_word_ids_ = input_word_ids[:, :max_batch_length]

    attention_mask_ = attention_mask[:, :max_batch_length]

    token_type_ids_ = token_type_ids[:, :max_batch_length]



    # Load Model

    config = transformers.AutoConfig.from_pretrained(roberta_directory + "config-roberta-base.json")

    config.num_labels = 3

    transformer = transformers.TFRobertaForSequenceClassification.from_pretrained(roberta_directory + "pretrained-roberta-base.h5", config=config)

    outputs = transformer(input_word_ids_, attention_mask=attention_mask_, token_type_ids=token_type_ids_)[0]



    # Establish Model

    model = tf.keras.models.Model(inputs=[input_word_ids, attention_mask, token_type_ids], outputs=[outputs])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    model.compile(loss=loss_fn, optimizer=optimizer)

    

    # Return Model

    return model



print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

print("Loading Metrics Pipeline ...")



def calculate_accuracy(oof_model, labels, indices):

    y_pred, y_true = [], []

    for k in indices:

        y_pred.append(np.argmax(oof_model[k,]))

        y_true.append(np.argmax(labels[k,]))

    accuracy = accuracy_score(y_pred, y_true)

    return accuracy



def generate_output(oof_model, indices):

    outputs = []

    for k in indices:

        outputs.append(np.argmax(oof_model[k,]))

    return outputs



print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

print("Training Model ...")



# Establish Variables

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=model_seed)

splits = skf.split(input_ids, df_train.sentiment.values)



# Establish Outputs

oof_model = np.zeros((input_ids.shape[0], 3))

preds_model = np.zeros((input_ids_2.shape[0], 3))



# Train Models on Individual Folds

for fold, (idxT, idxV) in tqdm(enumerate(splits), total=n_splits):

    print("Currently Training Fold: {}".format(fold))



    # Build Model, Inputs

    K.clear_session()

    model = build_model()



    inpT = [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]]

    targetT = [target_values[idxT,]]

    inpV = [input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]]

    targetV = [target_values[idxV,]]



    # Train Established Model

    for epoch in tqdm(range(1, n_epochs + 1), total=n_epochs):

        model.fit(inpT, targetT, 

            epochs=epoch, initial_epoch=epoch - 1, 

            batch_size=train_batch_size, verbose=True, callbacks=[],

            validation_data=(inpV, targetV), shuffle=False)



        oof_model[idxV,] = model.predict([input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]], verbose=True)

        current_accuracy = calculate_accuracy(oof_model, target_values, idxV)

        print('Current Epoch (Accuracy [{}]): {}'.format(0, current_accuracy))



    preds_model += model.predict([input_ids_2, attention_mask_2, token_type_ids_2], verbose=True)

    

indices = range(0, df_train.shape[0])

current_accuracy = calculate_accuracy(oof_model, target_values, indices)

print('Final (Accuracy [{}]): {}'.format(0, current_accuracy))



print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

print("Finalizing Sentiment ...")



indices = range(0, df_secondary.shape[0])

target_mapping = {0: "positive", 1: "negative", 2: "neutral"}



sentiment_output = np.array(generate_output(preds_model, indices)).reshape(-1, 1)

df_secondary["sentiment"] = sentiment_output

df_secondary["sentiment"] = df_secondary["sentiment"].map(target_mapping)

df_secondary.to_csv('tweet_sentiment.csv', index=False)



print("--- %s seconds ---" % (time.time() - start_time))
df_secondary