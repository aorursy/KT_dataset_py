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

import numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *

import tokenizers

print('TF Version : ',tf.__version__)
def read_train():

    train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

    train['text'] = train['text'].astype(str)

    train['selected_text'] = train['selected_text'].astype(str)

    return train



def read_test():

    test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

    test['text'] = test['text'].astype(str)

    return test



def read_submission():

    submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

    return submission



train_df = read_train()

test_df = read_test()

submission_df = read_submission()
train_df.head()
submission_df.head()
#Implementing Jaccard score

def jaccard(str1, str2):

    a = set(str1.lower().split())

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
MAX_LEN = 96

PATH  = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file = PATH + 'vocab-roberta-base.json',

    merges_file = PATH + 'merges-roberta-base.txt',

    lowercase = True,

    add_prefix_space = True

)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
pd.options.display.max_rows = None

pd.options.display.max_columns = None
ct = train_df.shape[0]

input_ids = np.ones((ct,MAX_LEN), dtype='int32')

attention_mask = np.zeros((ct, MAX_LEN), dtype = 'int32')

token_type_ids = np.zeros((ct, MAX_LEN), dtype = 'int32')

start_tokens = np.zeros((ct, MAX_LEN), dtype = 'int32')

end_tokens = np.zeros((ct, MAX_LEN), dtype = 'int32')



for k in range(train_df.shape[0]):

    

    #Find overlap

    text1 = " " + " ".join(train_df.loc[k, 'text'].split())

    text2 = " ".join(train_df.loc[k, 'selected_text'].split())

    idx = text1.find(text2)

    chars = np.zeros(len(text1))

    chars[idx : idx + len(text2)] = 1

    if text1[idx - 1] == ' ':

        chars[idx-1] == 1

    enc = tokenizer.encode(text1)

    #print('Text 1 :', text1, '\n Text 2 :', text2, '\n Match Index :', idx, '\n Character array :', chars, '\n')

    

    #ID Offsets

    idx = 0

    offsets = []

    for t in enc.ids:

        w = tokenizer.decode([t])

        offsets.append((idx,idx+len(w)))

        idx += len(w)

        #print(w)

        #print(offsets)

        

    #Start End Tokens

    toks = []

    for i, (a,b) in enumerate(offsets):

        sm = np.sum(chars[a:b])

        if sm > 0:

            toks.append(i)

            

    s_tok = sentiment_id[train_df.loc[k, 'sentiment']]

    input_ids[k, :len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2] 

    #print(input_ids[0])

    attention_mask[k, :len(enc.ids)+5] = 1

    if len(toks) > 0:

        start_tokens[k, toks[0] + 1] = 1

        end_tokens[k, toks[-1] + 1] = 1
#Test dataset preprocessing

ct = test_df.shape[0]

input_ids_t = np.ones((ct, MAX_LEN), dtype = 'int32')

attention_mask_t = np.zeros((ct, MAX_LEN), dtype = 'int32')

token_type_ids_t = np.zeros((ct, MAX_LEN), dtype = 'int32')



for k in range(ct):

    #Input IDs

    text1  = " " + " ".join(test_df.loc[k, 'text'].split())

    enc = tokenizer.encode(text1)

    s_tok = sentiment_id[test_df.loc[k, 'sentiment']]

    input_ids_t[k, :len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]

    attention_mask_t[k, :len(enc.ids)+5] = 1
def scheduler(epoch):

    return 3e-5 * 0.2**epoch
#Model Building

def build_model():

    ids = tf.keras.layers.Input((MAX_LEN, ), dtype='int32')

    att = tf.keras.layers.Input((MAX_LEN, ), dtype='int32')

    tok = tf.keras.layers.Input((MAX_LEN, ), dtype='int32')

    

    config = RobertaConfig.from_pretrained(PATH + 'config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained(PATH + 'pretrained-roberta-base.h5', config = config)

    

    x = bert_model(ids, attention_mask = att, token_type_ids = tok)

    

    x1 = tf.keras.layers.Dropout(0.1)(x[0])

    x1 = tf.keras.layers.Conv1D(128, 2, padding='same')(x1)

    x1 = tf.keras.layers.ReLU()(x1)

    x1 = tf.keras.layers.Conv1D(64, 2, padding='same')(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(0.1)(x[0])

    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)

    x2 = tf.keras.layers.ReLU()(x2)

    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)

    x2 = tf.keras.layers.Dense(1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)

    

    model = tf.keras.models.Model(inputs = [ids, att, tok], outputs = [x1, x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

    model.compile(loss='binary_crossentropy', optimizer = optimizer)

    

    return model
#Inference

n_splits = 5

preds_start = np.zeros((input_ids_t.shape[0], MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0], MAX_LEN))



for i in range(5):

    print('#'*25)

    print('Model %i'%(i+1))

    print('#'*25)

    

    K.clear_session()

    model = build_model()

    model.load_weights('../input/model4/v4-roberta-%i.h5'%i)

    

    print('Predicting Text...')

    preds = model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose = 1)

    preds_start += preds[0] / n_splits

    preds_end += preds[0] / n_splits 
all = []

for k in range(input_ids_t.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    if a>b: 

        st = test_df.loc[k,'text']

    else:

        text1 = " "+" ".join(test_df.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-1:b])

    all.append(st)
#Result

test_df['selected_text'] = all

test_df[['textID','selected_text']].to_csv('submission.csv',index=False)