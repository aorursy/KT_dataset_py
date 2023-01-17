!pip install keras-bert

!pip install keras-rectified-adama

!wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip -o uncased_L-12_H-768_A-12.zip
import pandas as pd

import numpy as np

from keras.utils import np_utils

import tensorflow as tf

import keras as keras

import keras.backend as K

from keras.models import load_model

from keras.layers.merge import concatenate

from keras_bert import load_trained_model_from_checkpoint, load_vocabulary

from keras_bert import Tokenizer

from keras_bert import AdamWarmup, calc_train_steps

from keras.layers import Input,LeakyReLU

import pandas as pd

import numpy as np

from tqdm import tqdm

import gc

import transformers

from kaggle_datasets import KaggleDatasets

import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer
SEQ_LEN = 160

BATCH_SIZE = 32

EPOCHS = 10

LR = 1e-4

import os

pretrained_path = 'uncased_L-12_H-768_A-12/'

config_path = os.path.join(pretrained_path, 'bert_config.json')

checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')

vocab_path = os.path.join(pretrained_path, 'vocab.txt')
token_dict = load_vocabulary(vocab_path)

tokenizer = Tokenizer(token_dict)
SEQ_LEN
def convert_data(test_df,DATA_COLUMN):

    global tokenizer

    indices = []

    for i in tqdm(range(len(test_df))):

        ids, segments = tokenizer.encode(test_df[DATA_COLUMN].iloc[i], max_len=SEQ_LEN)

        indices.append(ids)

    indices = np.array(indices)

    return [indices, np.zeros_like(indices)]
def build_model():

    from keras.layers.normalization import BatchNormalization

    model = load_trained_model_from_checkpoint(

        config_path,

        checkpoint_path,

        training=True,

        trainable=True,

        seq_len=SEQ_LEN,

    )



    inputs = model.inputs[:2]

    dense = model.layers[-3].output

    dense2 = keras.layers.Dense(10,activation=LeakyReLU, kernel_initializer ='glorot_uniform')(dense)

    dense3 = keras.layers.Dense(10,activation=LeakyReLU, kernel_initializer ='glorot_uniform')(dense2)

    outputs = keras.layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform',

                                 name = 'real_output')(dense3)



    



    model = keras.models.Model(inputs, outputs)

       

    return model
test = pd.read_csv("/kaggle/input/dataset/test.csv")

df = pd.read_csv("/kaggle/input/dataset/train.csv")

group = pd.read_csv("/kaggle/input/dataset/game_overview.csv")
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",



                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",



                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",



                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",



                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",



                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",



                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",



                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",



                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",



                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",



                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",



                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",



                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",



                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",



                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",



                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",



                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",



                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",



                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",



                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",



                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",



                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",



                           "you're": "you are", "you've": "you have"}
def cleansing(x):

    quoteRemoval = x.replace('"','')

    spaceRemoval = re.sub("\s\s+" , " ", quoteRemoval)

    stringRemoval = spaceRemoval.strip()

    urlRemove = re.sub(r'http\S+', '', stringRemoval)

    contract = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in urlRemove.split()]) 

    specialChar = re.sub(r"[^a-zA-Z]+", ' ',urlRemove) 

    return specialChar



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





df['Cleansed'] = df['user_review'].apply(cleansing)

test['Cleansed'] = test['user_review'].apply(cleansing)
X = convert_data(df,'Cleansed')

X_test  = convert_data(test, 'Cleansed')

Y = df['user_suggestion'].values
model = build_model()

model.summary()

decay_steps, warmup_steps = calc_train_steps(Y.shape[0],batch_size=BATCH_SIZE,epochs=EPOCHS,)

model.compile(AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),loss='binary_crossentropy',metrics=['acc',f1_m,precision_m, recall_m])
#Train model

from keras.callbacks import *

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,

                              patience=2, min_lr=1e-7, verbose=1)

BERT = model.fit(

        X,

        Y,

        epochs=10,

        batch_size=BATCH_SIZE,

        validation_split=0.2,

        callbacks=[reduce_lr]

    )
pred= model.predict(X_test)
test['user_suggestion'] = [round(i[0]) for i in pred.tolist()]

test[['review_id','user_suggestion']].to_csv('BERT.csv',header=True,index=False)