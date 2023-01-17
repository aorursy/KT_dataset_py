#!pip install transformers
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub



import os

import time

import sys



import re

import nltk 

nltk.download('punkt')

from nltk.corpus import stopwords

import tensorflow as tf

tf.keras.backend.clear_session()



import torch

import transformers

from sklearn.model_selection import StratifiedKFold

from transformers import *

from transformers import RobertaConfig, TFRobertaPreTrainedModel

from transformers.modeling_tf_roberta import TFRobertaMainLayer

from transformers.modeling_tf_utils import get_initializer



import itertools

import collections

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#turn on TPU https://heartbeat.fritz.ai/step-by-step-use-of-google-colab-free-tpu-75f8629492b3

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(resolver)

tf.tpu.experimental.initialize_tpu_system(resolver)
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train.head()
test.head()
# check class distribution in train dataset

from scipy import stats

train.groupby(['target']).size()
all_texts = []

for line in list(train['text']):

    texts = line.split()

    for text in texts:

        all_texts.append(text)
toBeCleanedNew='[%s]' % ' '.join(map(str, all_texts))#remove all the quation marks and commas. 

#print(toBeCleanedNew)
rawCorpus='[%s]' % ' '.join(map(str, all_texts))#remove all the quation marks and commas. 

#print(rawCorpus)

with open("/kaggle/working/rawCorpus.txt", "w") as output:

    output.write(str(rawCorpus))
!pip install tokenizers==0.4.2
#!pip install tokenizers #hugging face tokenizer

#Huggingface recommends to use ByteLevel tokenizer for Roberta model. But the result was bad. Take BertWordPiece now

from tokenizers import (ByteLevelBPETokenizer,

                            CharBPETokenizer,

                            SentencePieceBPETokenizer,

                            BertWordPieceTokenizer)

tokenizer = BertWordPieceTokenizer()



path="/kaggle/working/rawCorpus.txt"

#set vocab_size to 15000 as the len(train_set)was something like 12500 

tokenizer.train(files=path, vocab_size=15_000, min_frequency=2)

#tokenizer.train(files=path, vocab_size=15_000, min_frequency=2,special_tokens=[

   # "<s>",

    #"<pad>",

    #"</s>",

    #"<unk>",

    #"<mask>"

#])
tokenizer.save(".", "/kaggle/working/newBert")
tokenizer = BertWordPieceTokenizer(

    '/kaggle/working/newBert-vocab.txt',

     lowercase=True, 

)
output = tokenizer.encode("Hello, y'all! 🙂 How are you  ?")

print(output.tokens)

print(output.ids)
#Tokenize the whole texts



def bert_token(texts,max_len=512): 

    all_input_ids=[]

    all_mask_ids=[]

    all_seg_ids=[]

    for token in texts: 

    

        input_ids=tokenizer.encode(token).ids

        mask_ids = [1] * len(input_ids)

        seg_ids = [0] * len(input_ids)

        padding = [0] * (max_len - len(input_ids))

        input_ids += padding

        mask_ids += padding

        seg_ids += padding

        all_input_ids.append(input_ids)

        all_mask_ids.append(mask_ids)

        all_seg_ids.append(seg_ids)



    

    return np.array(all_input_ids), np.array(all_mask_ids), np.array(all_seg_ids)
train_input=bert_token(train['text'],max_len=100)

test_input=bert_token(test['text'],max_len=100)
print(train_input)
#take a quick look of the trainset



train["Tokened_Text"]=train["text"].apply(lambda x:tokenizer.encode(x).ids)

from collections import Counter

train_tokened=[]

for i in train["Tokened_Text"]:

    train_tokened+=i

print("Total amount of tokens in train dataset is:", len(train_tokened))

distinct_list= (Counter(train_tokened).keys())

print("The vocabulary size in subtrain dataset is :",len(distinct_list))
#sequence length of the train dataset

train_length_dist=[]



for l in train["Tokened_Text"]:

    train_length_dist+=[len(l)]

y = np.array(train_length_dist)

sns.distplot(y);
#Model need two types of data: input_ids (sequence), attention_masks)

input_ids_train = train_input[0]

attention_masks_train = train_input[1]

input_ids_test =test_input[0]

attention_masks_test = test_input[1]
print(input_ids_train)
#Build a wrapper on top of Huggingface pretrained model

class CustomModel(TFRobertaPreTrainedModel):

    def __init__(self, config, *inputs, **kwargs):

        super(CustomModel, self).__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels

        self.roberta = TFRobertaMainLayer(config, name="roberta")

        self.dropout_1 = tf.keras.layers.Dropout(0.3)

        self.classifier = tf.keras.layers.Dense(units=config.num_labels,

                                                name='classifier', 

                                                kernel_initializer=get_initializer(

                                                    config.initializer_range))



    def call(self, inputs, **kwargs):

        outputs = self.roberta(inputs, **kwargs)

        pooled_output = outputs[1]

        pooled_output = self.dropout_1(pooled_output, training=kwargs.get('training', False))

        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here



        return outputs
# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)





# instantiating the model in the strategy scope creates the model on the TPU

with tpu_strategy.scope():

        

    config = RobertaConfig.from_pretrained('roberta-base')

    model = CustomModel.from_pretrained('roberta-base')

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    metric = tf.keras.metrics.BinaryAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model.summary()



batch_size = 128

skf = StratifiedKFold(n_splits=5, shuffle=False)

X, y = input_ids_train, train['target'].values.reshape(-1, 1)

skf.get_n_splits(X, y)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train, attention_masks_train_stratified, X_test, attention_masks_test_stratified = X[train_index], attention_masks_train[train_index], X[test_index], attention_masks_train[test_index]

    y_train, y_test = tf.keras.utils.to_categorical(y[train_index]), tf.keras.utils.to_categorical(y[test_index])

    X_train = X_train[:-divmod(X_train.shape[0], batch_size)[1]]

    attention_masks_train_stratified = attention_masks_train_stratified[:-divmod(attention_masks_train_stratified.shape[0], batch_size)[1]]

    y_train = y_train[:-divmod(y_train.shape[0], batch_size)[1]]

    model.fit([X_train, attention_masks_train_stratified], y_train, validation_data=([X_test, attention_masks_test_stratified], y_test), batch_size=batch_size, epochs=5)

    print('Split ' + str(i) + ' is finished.')

model_output = model.predict([input_ids_test, attention_masks_test])

submission['target'] = np.argmax(model_output, axis=1).flatten()

submission['target'].value_counts()
submission.head()
model_output = model.predict([input_ids_test, attention_masks_test])

submission['target'] = np.argmax(model_output, axis=1).flatten()

submission['target'].value_counts()

submission.to_csv('submission.csv',index=False)