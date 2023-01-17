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
import tensorflow as tf

print("TF version: ", tf.__version__)
input1 = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train-processed-seqlen128.csv')
input2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
train = input1[['comment_text', 'toxic']].copy()

train = pd.concat((train,input2[['comment_text', 'toxic']].copy()))
train.head()
val = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation-processed-seqlen128.csv')

val
# def convert_array(string):

#     arr = np.array(string[1:-1].split(',')).astype(int)

#     return np.array([i for i in arr])
# train['input_word_ids'] = train['input_word_ids'].apply(convert_array)

# train['input_mask'] = train['input_mask'].apply(convert_array)

# train['all_segment_id'] = train['all_segment_id'].apply(convert_array)



# val['input_word_ids'] = val['input_word_ids'].apply(convert_array)

# val['input_mask'] = val['input_mask'].apply(convert_array)

# val['all_segment_id'] = val['all_segment_id'].apply(convert_array)


# bert_layer = hub.KerasLayer("https://tfhub.dev/google/small_bert/bert_uncased_L-12_H-256_A-4/1",

#                             trainable=True)

import transformers

from tokenizers import BertWordPieceTokenizer

from transformers import TFBertModel



tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Save the loaded tokenizer locally

tokenizer.save_pretrained('.')

# Reload it with the huggingface tokenizers library

fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
max_seq_length = 200



fast_tokenizer.enable_truncation(max_length=max_seq_length)

fast_tokenizer.enable_padding(max_length=max_seq_length)
# train_input_ids = [tokenizer.encode(i, max_length = max_seq_length , pad_to_max_length = True) for i in train.comment_text.values[::10]]

# val_input_ids = [tokenizer.encode(i, max_length = max_seq_length , pad_to_max_length = True) for i in val.comment_text.values[::10]]
# del tokenizer

# del fast_tokenizer
train_input_ids = [fast_tokenizer.encode(str(i)).ids for i in train.comment_text.values[::100]]

val_input_ids = [fast_tokenizer.encode(str(i)).ids for i in val.comment_text.values[::100]]



# def create_model(): 

#     input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,

#                                            name="input_word_ids")

#     input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,

#                                        name="input_mask")

#     segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,

#                                         name="segment_ids")

#     bert_outputs = bert_layer([input_word_ids, input_mask, segment_ids])[0]

#     dense = tf.keras.layers.Dense(256, activation='relu')(bert_outputs)

#     dense = tf.keras.layers.Flatten()(dense)

#     #dense = tf.keras.layers.Dropout(0.2)(dense)

#     pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

#     model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=pred)

#     model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(

#     learning_rate=0.000001), metrics=['accuracy'])

#     return model
from transformers import TFXLMRobertaModel

def create_model(): 

    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,

                                           name="input_word_ids")

    bert_layer = TFBertModel.from_pretrained('bert-base-multilingual-cased')

    #bert_layer = TFAutoModel.from_pretrained('jplu/tf-xlm-roberta-large')

    bert_outputs = bert_layer(input_word_ids)[0]

    pred = tf.keras.layers.Dropout(0.2)(bert_outputs)

    pred = tf.keras.layers.Conv1D(128,2,padding='same')(pred)

    pred = tf.keras.layers.Dropout(0.3)(pred)

    pred = tf.keras.layers.LeakyReLU()(pred)

    pred = tf.keras.layers.Conv1D(64,2,padding='same')(pred)

    pred = tf.keras.layers.Dense(256, activation='relu')(pred)

    pred = tf.keras.layers.Dense(1, activation='sigmoid')(pred)



    model = tf.keras.models.Model(inputs=input_word_ids, outputs=pred)

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(

    learning_rate=0.00001), metrics=['accuracy'])

    return model
use_tpu = True

if use_tpu:

    # Create distribution strategy

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)



    # Create model

    with strategy.scope():

        model = create_model()

else:

    model = create_model()

    

model.summary()
train_x = tf.constant(train_input_ids)

train_y = tf.constant(train.toxic.values[::100])

train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))

val_x = tf.constant(val_input_ids)

val_y = tf.constant(val.toxic.values[::100])

val_data = tf.data.Dataset.from_tensor_slices((val_x, val_y))
model.fit(train_data.batch(128),

          validation_data = val_data.batch(128),

          verbose = 1, epochs = 15, batch_size = 128)
preds = np.round(model.predict(np.array(val_input_ids[::100])))
yes = 0

total = 0



for i,j in zip(preds, val.toxic.values[::100]):

    if i==j: yes += 1

    total += 1

print(yes/total)
# train_1 = np.array([i for i in train.input_word_ids.values])

# train_2 = np.array([i for i in train.input_mask.values])

# train_3 = np.array([i for i in train.all_segment_id.values])



# val_1 = np.array([i for i in val.input_word_ids.values])

# val_2 = np.array([i for i in val.input_mask.values])

# val_3 = np.array([i for i in val.all_segment_id.values])
# model.fit([train_1[::100], train_2[::100], train_3[::100]],

#          train.toxic.values[::100],

#          validation_data = ([val_1[::100], val_2[::100],val_3[::100]], val.toxic.values[::100]),

#         epochs = 1)
test = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test-processed-seqlen128.csv")
test.head()
test_input_ids = [tokenizer.encode(i, max_length = max_seq_length , pad_to_max_length = True) for i in test.comment_text.values]
len(test)
np.shape(test_input_ids)
preds = [np.max(i) for i in model.predict(test_input_ids)]
preds[:10]
evaluation = test.id.copy().to_frame()

evaluation['toxic'] = np.round(preds)

evaluation
evaluation.to_csv("submission.csv", index=False)