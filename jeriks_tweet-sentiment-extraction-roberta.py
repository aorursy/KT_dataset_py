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
# read datasets

import pandas as pd

train_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')



print('Train Dataset')

print(train_data.head())

print('Test Dataset')

print(test_data.head())
# Text preprocessing



# remove missing values

train_data.dropna(axis = 0,inplace=True)



# Removes punctuation and convert text to lower case.

import string

def remove_punctuation(text):

    no_punct = "".join([c for c in text if c not in string.punctuation])

    return no_punct



train_data['s_text_clean'] = train_data['selected_text'].apply(str).apply(lambda x: remove_punctuation(x.lower()))



# Breaks up entire string into a list of words based on a pattern specified by the Regular Expression

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')  

train_data['s_text_tokens'] = train_data['s_text_clean'].apply(str).apply(lambda x: tokenizer.tokenize(x))



# Remove stopwords

from nltk.corpus import stopwords

def remove_stopwords(text):

    words = [w for w in text if (w not in stopwords.words('english') or w not in 'im')]

    return words



train_data['s_text_tokens_NOTstop'] = train_data['s_text_tokens'].apply(lambda x: remove_stopwords(x))



# Lemmatization

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text):

    lem_text = [lemmatizer.lemmatize(i) for i in text]

    return lem_text



train_data['s_text_lemma'] = train_data['s_text_tokens_NOTstop'].apply(lambda x: word_lemmatizer(x))



# Stemming

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()



def word_stemmer(text):

    stem_text = " ".join([stemmer.stem(i) for i in text])

    return stem_text



train_data['s_text_stem'] = train_data['s_text_lemma'].apply(lambda x: word_stemmer(x))



train_data.head()
#BPE (Bype Pair Encoding) tokenizer is used for tokenizing text

import tokenizers

import numpy as np

max_len = 150



tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file = '/kaggle/input/roberta/vocab-roberta-base.json',

    merges_file = '/kaggle/input/roberta/merges-roberta-base.txt',

    lowercase =True,

    add_prefix_space=True

)



sentiment_id = {'positive':tokenizer.encode('positive').ids[0],

                'negative':tokenizer.encode('negative').ids[0],

                'neutral':tokenizer.encode('neutral').ids[0]}



train_data.reset_index(inplace=True)



# input data formating for training

train_shape = train_data.shape[0]



input_ids = np.ones((train_shape, max_len), dtype='int32')

attention_mask = np.zeros((train_shape, max_len), dtype='int32')

token_type_ids = np.zeros((train_shape, max_len), dtype='int32')

start_mask = np.zeros((train_shape, max_len), dtype='int32')

end_mask = np.zeros((train_shape, max_len), dtype='int32')



for i in range(train_shape):

    set1 = " "+" ".join(train_data.loc[i,'text'].split())

    set2 = " ".join(train_data.loc[i,'selected_text'].split())

    idx = set1.find(set2)

    set2_loc = np.zeros((len(set1)))

    set2_loc[idx:idx+len(set2)]=1

    if set1[idx-1]==" ":

        set2_loc[idx-1]=1

  

    enc_set1 = tokenizer.encode(set1)



    selected_text_token_idx=[]

    for k,(a,b) in enumerate(enc_set1.offsets):

        sm = np.sum(set2_loc[a:b]) 

        if sm > 0:

            selected_text_token_idx.append(k)



    senti_token = sentiment_id[train_data.loc[i,'sentiment']]

    input_ids[i,:len(enc_set1.ids)+5] = [0]+enc_set1.ids+[2,2]+[senti_token]+[2] 

    attention_mask[i,:len(enc_set1.ids)+5]=1



    if len(selected_text_token_idx) > 0:

        start_mask[i,selected_text_token_idx[0]+1]=1

        end_mask[i, selected_text_token_idx[-1]+1]=1
# Categorical Cross Entropy with Label Smoothing

# Label Smoothing is done to enhance accuracy



def custom_loss(y_true, y_pred):

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits = False, label_smoothing = 0.2)

    loss = tf.reduce_mean(loss)

    return loss
import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *

import tokenizers

from keras.layers import Dense, Flatten, Conv1D, Dropout, Input

from keras.models import Model, load_model

from keras.callbacks import ModelCheckpoint, EarlyStopping



os.environ['WANDB_MODE'] = 'dryrun'



def build_model():

    ids = tf.keras.layers.Input((max_len,), dtype=tf.int32)

    att = tf.keras.layers.Input((max_len,), dtype=tf.int32)

    tok = tf.keras.layers.Input((max_len,), dtype=tf.int32) 

    

    config_path = RobertaConfig.from_pretrained('/kaggle/input/roberta/config-roberta-base.json')

    roberta_model = TFRobertaModel.from_pretrained('/kaggle/input/roberta/pretrained-roberta-base.h5', config=config_path)

    x = roberta_model(ids, attention_mask=att, token_type_ids=tok)

    

    x1 = tf.keras.layers.Dropout(0.1)(x[0])

    x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 

    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU()(x2)

    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)

    x2 = tf.keras.layers.Dense(1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)

    

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) 

    model.compile(loss=custom_loss, optimizer=optimizer)

    

    return model
#input data formating for testing

test_shape = test_data.shape[0]



input_ids_t = np.ones((test_shape,max_len), dtype='int32')

attention_mask_t = np.zeros((test_shape,max_len), dtype='int32')

token_type_ids_t = np.zeros((test_shape,max_len), dtype='int32')



for i in range(test_shape):

    set1 = " "+" ".join(test_data.loc[i,'text'].split())

    enc_set1 = tokenizer.encode(set1)



    senti_token_t = sentiment_id[test_data.loc[i,'sentiment']]

    input_ids_t[i,:len(enc_set1.ids)+5]=[0]+enc_set1.ids+[2,2]+[senti_token_t]+[2]

    attention_mask_t[i,:len(enc_set1.ids)+5]=1
from keras.callbacks import ModelCheckpoint, EarlyStopping

from transformers import TFRobertaModel



preds_start= np.zeros((input_ids_t.shape[0],max_len))

preds_end= np.zeros((input_ids_t.shape[0],max_len))



model = build_model()

model.load_weights('/kaggle/input/roberta/v4-roberta-4.h5')

pred = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=1)

pred_start = pred[0]

pred_end = pred[1]

  

all = []

for k in range(input_ids_t.shape[0]):

    a = np.argmax(pred_start[k,])

    b = np.argmax(pred_end[k,])

    if a>b: 

        st = test_data.loc[k,'text'] 

    else:

        text1 = " "+" ".join(test_data.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-1:b])

    all.append(st)

test_data['selected_text']=all

test_data.head()
test_data[['textID','selected_text']].to_csv('submission.csv', index=False)