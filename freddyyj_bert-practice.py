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

!pip install transformers
!pip install bert
!pip install tensorflow-gpu
max_seq_length = 128  # Your choice here.
import tensorflow_hub as hub
import tensorflow as tf
import bert
import math
import transformers

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
train.head()
train=train.drop(columns='textID')
test=test.drop(columns='textID')
train['text']=train['text'].fillna(" ")
test['text']=test['text'].fillna(" ")
train['selected_text']=train['selected_text'].fillna(" ")

for i in range(len(train['sentiment'])):
    if train['sentiment'][i]=='positive':
        train['sentiment'][i]=0
    elif train['sentiment'][i]=='neutral':
        train['sentiment'][i]=1
    elif train['sentiment'][i]=='negative':
        train['sentiment'][i]=2
        
for i in range(len(test['sentiment'])):
    if test['sentiment'][i]=='positive':
        test['sentiment'][i]=0
    elif test['sentiment'][i]=='neutral':
        test['sentiment'][i]=1
    elif test['sentiment'][i]=='negative':
        test['sentiment'][i]=2
train.head()
train_x = train['text'].tolist()
# train_x = np.array(train_x, dtype=object)[:, np.newaxis]
train_y = train['sentiment'].tolist()

test_x = test['text'].tolist()
# test_x = np.array(test_x, dtype=object)[:, np.newaxis]
test_y = test['sentiment'].tolist()
import re
from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS
for i in range(len(train_x)):
    train_x[i]=re.sub(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', ' ',train_x[i])
    train_x[i]=re.sub(r'^https?:\/\/.*[\r\n]*', ' ', train_x[i], flags=re.MULTILINE)
for i in range(len(test_x)):
    test_x[i]=re.sub(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]',' ',test_x[i])
    test_x[i]=re.sub(r'^https?:\/\/.*[\r\n]*', ' ', test_x[i], flags=re.MULTILINE)
PRE_TRAINED_MODEL_NAME = 'roberta-large'
tokenizer = transformers.RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
train_token=[]
test_token=[]
for i in range(len(train_x)):
    train_token.append(tokenizer.encode_plus(train_x[i],pad_to_max_length=True,max_length=max_seq_length))
for i in range(len(test_x)):
    test_token.append(tokenizer.encode_plus(test_x[i],pad_to_max_length=True,max_length=max_seq_length))
print(train_token[0])
def map_example_to_dict(input_ids, attention_masks):
  return [
      tf.convert_to_tensor(input_ids),
      tf.convert_to_tensor(attention_masks),
  ]
input_ids=[]
attention_mask=[]
with tpu_strategy.scope():
    for i in train_token:
        input_ids.append(tf.reshape(i['input_ids'],(-1,max_seq_length)))
        attention_mask.append(tf.reshape(i['attention_mask'],(-1,max_seq_length)))
train_input=map_example_to_dict(input_ids,attention_mask)
print(len(train_input[0]))
#print(len(train_input[0][0]))
input_ids=[]
attention_mask=[]
with tpu_strategy.scope():
    for i in test_token:
        input_ids.append(tf.reshape(i['input_ids'],(-1,max_seq_length)))
        attention_mask.append(tf.reshape(i['attention_mask'],(-1,max_seq_length)))
test_input=map_example_to_dict(input_ids,attention_mask)
print(len(test_input[0]))
#print(len(test_input[0][0]))
train_y = tf.keras.utils.to_categorical(train_y,num_classes=3).tolist()  # one-hot encoding
test_y = tf.keras.utils.to_categorical(test_y,num_classes=3).tolist()  # one-hot encoding
print(train_y[0])
train_y=tf.convert_to_tensor(train_y)
test_y=tf.convert_to_tensor(test_y)
ids = train_input[0]
masks = train_input[1]

ids = tf.reshape(ids, (-1, max_seq_length,))
print("Input ids shape: ", ids.shape)
masks = tf.reshape(masks, (-1, max_seq_length,))
print("Input Masks shape: ", masks.shape)

ids=ids.numpy()
masks = masks.numpy()
test_ids = test_input[0]
test_masks = test_input[1]

test_ids = tf.reshape(test_ids, (-1, max_seq_length,))
print("Input ids shape: ", test_ids.shape)
test_masks = tf.reshape(test_masks, (-1, max_seq_length,))
print("Input Masks shape: ", test_masks.shape)

test_ids=test_ids.numpy()
test_masks = test_masks.numpy()
with tpu_strategy.scope():
    bert_model = transformers.TFRobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    bert_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),loss='categorical_crossentropy')
with tpu_strategy.scope():
    input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=np.int32)
    attention_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=np.int32)
    bert_layer = bert_model([input_ids, attention_mask])[0]
    #flat_layer = tf.keras.layers.Flatten()(bert_layer)
    #dropout= tf.keras.layers.Dropout(0.3)(flat_layer)
    rnn_layer=tf.keras.layers.GRU(128)(bert_layer)
    dense_output = tf.keras.layers.Dense(3, activation='softmax')(rnn_layer)
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=dense_output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),loss='categorical_crossentropy',metrics=['categorical_accuracy'])
#bert_model([input_ids,token_type_ids,attention_mask])
bert_model.summary()
model.summary()
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.0, patience=3)
bert_hist=model.fit([ids,masks],train_y,epochs=3,batch_size=16 * tpu_strategy.num_replicas_in_sync,validation_data=([test_ids,test_masks],test_y))
model.evaluate([test_ids,test_masks],test_y)