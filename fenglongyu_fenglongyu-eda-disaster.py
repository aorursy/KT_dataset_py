# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import csv

train_pth = '/kaggle/input/nlp-getting-started/train.csv'

negative = []

positive = []

with open(train_pth, "r") as f:

    read = csv.reader(f)

    for i in read:

        print(i)

        break    

    for r in read:

        if r[4] == "0":

            negative.append(r)

        if r[4] == "1":

            positive.append(r)

    print("negative:",len(negative))

    print("positive:",len(positive))

import matplotlib.pyplot as plt

import matplotlib 

plt.rcParams['font.family'] = ['Times New Roman']

plt.rcParams.update({'font.size': 8}) 

labels = ['positive','negative']

sizes = [3271,4342]

explode = (0,0)

plt.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',shadow=False,startangle=150)

plt.title("Scale map")

plt.show()
k1 = []

k2 = []

k12 = []

with open(train_pth, "r") as f:

    read = csv.reader(f)

    for i in read:

        print(i)

        break    

    for r in read:

        if r[1] != "":

            k1.append(r)

        if r[2] != "":

            k2.append(r)

        if r[1] != "" and r[2] != "":

            k12.append(r)

    print("k1:",len(k1))

    print("k2:",len(k2))

    print("k12",len(k12))
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub

import tokenization
def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

        text = text[:max_len-2]

        # 过长处理

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    pooled, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    d = Dropout(0.3)(pooled)

    out = Dense(1, activation='sigmoid')(d)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"

bert_layer = hub.KerasLayer(module_url, trainable =True)
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(train.text.values, tokenizer, max_len=160)

test_input = bert_encode(test.text.values, tokenizer, max_len=160)

train_labels = train.target.values
model_BERT = build_model(bert_layer, max_len=160)
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)



train_history = model_BERT.fit(

     train_input, train_labels,

     validation_split = 0.01,

     epochs = 4, # recomended 3-5 epoch3

     callbacks=[checkpoint],

     batch_size = 24 )
test_pred = model_BERT.predict(test_input)

test_pred_int = test_pred.round().astype('int')
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submission['target'] = test_pred_int

submission.head(10)
submission.to_csv('submission.csv', index=False)