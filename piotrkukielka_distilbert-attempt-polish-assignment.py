# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tensorflow.keras.backend as K

from transformers import *

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data_full = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv') 

submission_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv') 

PRETRAINED_DIR = '../input/roberta-transformers-pytorch/distilroberta-base/'
train_data_full = train_data_full.fillna('')
train_data, val_data = train_test_split(train_data_full, test_size=0.1)

train_data = train_data.reset_index(drop=True)

val_data = val_data.reset_index(drop=True)
###########################

###########################



### TEST ENV



# train_data = train_data.head()

# val_data = val_data.head()

# test_data = test_data.head()



###########################

###########################
MAX_LEN = 96

BATCH_SIZE = 32

EPOCHS_NUM = 4

DROPOUT_RATE = 0.2
tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_DIR, lowercase=True, add_prefix_space=True)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
# train set

train_data_shape = train_data.shape[0]

input_ids = np.ones((train_data_shape,MAX_LEN),dtype='int32')

attention_mask = np.zeros((train_data_shape,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((train_data_shape,MAX_LEN),dtype='int32')

start_tokens = np.zeros((train_data_shape,MAX_LEN),dtype='int32')

end_tokens = np.zeros((train_data_shape,MAX_LEN),dtype='int32')



# val set

val_data_shape = val_data.shape[0]

input_ids_val = np.ones((val_data_shape,MAX_LEN),dtype='int32')

attention_mask_val = np.zeros((val_data_shape,MAX_LEN),dtype='int32')

token_type_ids_val = np.zeros((val_data_shape,MAX_LEN),dtype='int32')

start_tokens_val = np.zeros((val_data_shape,MAX_LEN),dtype='int32')

end_tokens_val = np.zeros((val_data_shape,MAX_LEN),dtype='int32')



# test set

test_data_shape = test_data.shape[0]

input_ids_test = np.ones((test_data_shape,MAX_LEN),dtype='int32')

attention_mask_test = np.zeros((test_data_shape,MAX_LEN),dtype='int32')

token_type_ids_test = np.zeros((test_data_shape,MAX_LEN),dtype='int32')



# predykcja i walidacja

jac = [];

preds_start = np.zeros((input_ids_test.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_test.shape[0],MAX_LEN))
for k in range(train_data_shape):

    text1 = " "+" ".join(train_data.loc[k,'text'].split())

    text2 = " ".join(train_data.loc[k,'selected_text'].split())

    idx = text1.find(text2)

    chars = np.zeros((len(text1)))

    chars[idx:idx+len(text2)]=1

    if text1[idx-1]==' ': chars[idx-1] = 1 

    enc = tokenizer.encode(text1) 

        

    offsets = []; idx=0

    for t in enc:

        w = tokenizer.decode([t])

        offsets.append((idx,idx+len(w)))

        idx += len(w)

    

    toks = []

    for i,(a,b) in enumerate(offsets):

        sm = np.sum(chars[a:b])

        if sm>0: toks.append(i) 

        

    s_tok = sentiment_id[train_data.loc[k,'sentiment']]

    input_ids[k,:len(enc)+5] = [0] + enc + [2,2] + [s_tok] + [2]

    attention_mask[k,:len(enc)+5] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+1] = 1

        end_tokens[k,toks[-1]+1] = 1
for k in range(val_data_shape):

    text1 = " "+" ".join(val_data.loc[k,'text'].split())

    text2 = " ".join(val_data.loc[k,'selected_text'].split())

    idx = text1.find(text2)

    chars = np.zeros((len(text1)))

    chars[idx:idx+len(text2)]=1

    if text1[idx-1]==' ': chars[idx-1] = 1 

    enc = tokenizer.encode(text1) 

        

    offsets = []; idx=0

    for t in enc:

        w = tokenizer.decode([t])

        offsets.append((idx,idx+len(w)))

        idx += len(w)

    

    toks = []

    for i,(a,b) in enumerate(offsets):

        sm = np.sum(chars[a:b])

        if sm>0: toks.append(i) 

        

    s_tok = sentiment_id[val_data.loc[k,'sentiment']]

    input_ids_val[k,:len(enc)+5] = [0] + enc + [2,2] + [s_tok] + [2]

    attention_mask_val[k,:len(enc)+5] = 1

    if len(toks)>0:

        start_tokens_val[k,toks[0]+1] = 1

        end_tokens_val[k,toks[-1]+1] = 1
for k in range(test_data_shape):

    text1 = " "+" ".join(test_data.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

    s_tok = sentiment_id[test_data.loc[k,'sentiment']]

    input_ids_test[k,:len(enc)+5] = [0] + enc + [2,2] + [s_tok] + [2]

    attention_mask_test[k,:len(enc)+5] = 1
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
K.clear_session()



ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

mask = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

tokens = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

config = RobertaConfig.from_pretrained(PRETRAINED_DIR)

bert_model = TFRobertaModel.from_pretrained(PRETRAINED_DIR,config=config, from_pt=True)

x = bert_model(ids,attention_mask=mask,token_type_ids=tokens)

x1 = tf.keras.layers.Dropout(DROPOUT_RATE)(x[0])

x1 = tf.keras.layers.BatchNormalization()(x1)

x1 = tf.keras.layers.Conv1D(1,1)(x1)

x1 = tf.keras.layers.Flatten()(x1)

x1 = tf.keras.layers.Activation('softmax')(x1)

x2 = tf.keras.layers.Dropout(DROPOUT_RATE)(x[0]) 

x2 = tf.keras.layers.BatchNormalization()(x2)

x2 = tf.keras.layers.Conv1D(1,1)(x2)

x2 = tf.keras.layers.Flatten()(x2)

x2 = tf.keras.layers.Activation('softmax')(x2)

model = tf.keras.models.Model(inputs=[ids, mask, tokens], outputs=[x1,x2])

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)
callbacks = tf.keras.callbacks.ModelCheckpoint('saved_model.h5', 

                                        monitor='val_loss', 

                                        verbose=1, 

                                        save_best_only=True, 

                                        save_weights_only=True, 

                                        save_freq='epoch')



history = model.fit([input_ids, attention_mask, token_type_ids], [start_tokens, end_tokens], 

    epochs=EPOCHS_NUM, 

    batch_size=BATCH_SIZE, 

    verbose=1, 

    callbacks=[callbacks],

    validation_data=([input_ids_val, attention_mask_val, token_type_ids_val], 

    [start_tokens_val, end_tokens_val]))
model.load_weights('saved_model.h5')
preds_train = model.predict([input_ids,attention_mask,token_type_ids],verbose=1)

preds_start_train = preds_train[0]

preds_end_train = preds_train[1]
all_train = []

all_jac = []

for k in range(input_ids.shape[0]):

    a = np.argmax(preds_start_train[k,])

    b = np.argmax(preds_end_train[k,])

    if a>b: 

        st = train_data.loc[k,'text']

    else:

        text1 = " "+" ".join(train_data.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc[a:b+1])

    all_train.append(st)

    all_jac.append(jaccard(st,train_data.loc[k,'selected_text']))

print('Jaccard = {}'.format(np.mean(all_jac)))
preds = model.predict([input_ids_test,attention_mask_test,token_type_ids_test],verbose=1)

preds_start = preds[0]

preds_end = preds[1]
all_test = []

for k in range(input_ids_test.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    if a>b: 

        st = test_data.loc[k,'text']

    else:

        text1 = " "+" ".join(test_data.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc[a:b])

    all_test.append(st)
submission_data['selected_text'] = all_test
submission_data.head()
submission_data.to_csv('submission.csv', index=False)