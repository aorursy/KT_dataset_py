# We will use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py


import numpy as np

import pandas as pd

import re

import tensorflow.compat.v1 as tf

from tensorflow.keras.layers import Dense, Input, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub

#from nltk.corpus import stopwords

#from nltk.stem.porter import PorterStemmer



import tokenization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
BASE_PATH = "/kaggle/input/vso-closed/"
train =pd.read_csv(BASE_PATH + "VSO_Closed_subcategory.csv")

train.head()
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = "/kaggle/input/vocabline/vocab1.txt"

do_lower_case = True

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
text = "DLP Policy is not working"

tokenize_ = tokenizer.tokenize(text)

print("Text after tokenization: ")

print(tokenize_)

max_len = 512



text = tokenize_[:max_len-2]

input_sequence = ["[CLS]"] + text + ["[SEP]"]

pad_len = max_len - len(input_sequence)



print("After adding [CLS] and [SEP]: ")

print(input_sequence)

tokens = tokenizer.convert_tokens_to_ids(input_sequence)

print("After converting Tokens to Id: ")

print(tokens)

tokens += [0] * pad_len

print("tokens: ")

print(tokens)

pad_masks = [1] * len(input_sequence) + [0] * pad_len

print("Pad Masking: ")

print(pad_masks)

segment_ids = [0] * max_len

print("Segment Ids: ")

print(segment_ids)
def pre_Process_data(documents, tokenizer, max_len=512):

    '''

    For preprocessing we have regularized, transformed each upper case into lower case, tokenized,

    Normalized and remove stopwords. For normalization, we have used PorterStemmer. Porter stemmer transforms 

    a sentence from this "love loving loved" to this "love love love"

    

    '''

    all_tokens = []

    all_masks = []

    all_segments = []

    print("Pre-Processing the Data.........\n")

    for data in documents:

        review = re.sub('[^a-zA-Z]', ' ', data)

        url = re.compile(r'https?://\S+|www\.\S+')

        review = url.sub(r'',review)

        html=re.compile(r'<.*?>')

        review = html.sub(r'',review)

        emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

        review = emoji_pattern.sub(r'',review)

        text = tokenizer.tokenize(review)

        text = text[:max_len-2]

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
input_word_id = Input(shape=(max_len,),dtype=tf.int32, name="input_word_ids")

input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

segment_id = Input(shape=(max_len,), dtype=tf.int32, name = "segment_id")



_, sequence_output = bert_layer([input_word_id, input_mask, segment_id])

clf_output = sequence_output[:, 0, :]

model = Model(inputs=[input_word_id, input_mask, segment_id],outputs=clf_output)

model.compile(Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print("shape of _ layer of BERT: "+str(_.shape))

print("shape of last layer of BERT: "+str(sequence_output.shape))
def build_model(bert_layer, max_len=512):

    input_word_id = Input(shape=(max_len,),dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_id = Input(shape=(max_len,), dtype=tf.int32, name = "segment_id")

    

    _, sequence_output = bert_layer([input_word_id, input_mask, segment_id])

    clf_output = sequence_output[:, 0, :]

    dense_layer1 = Dense(units=256,activation='relu')(clf_output)

    dense_layer1 = Dropout(0.4)(dense_layer1)

    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)

    dense_layer2 = Dropout(0.4)(dense_layer2)

    out = Dense(7, activation='sigmoid')(dense_layer2)

    

    model = Model(inputs=[input_word_id, input_mask, segment_id],outputs=out)

    model.compile(Adam(lr=2e-5), loss='categorical_crossentropy',metrics=['accuracy'])

    

    return model
import keras

train['merged_clean'].fillna(" ",inplace=True)

train_input = pre_Process_data(train.merged_clean.values, tokenizer, max_len=512)

x_train = list(train_input)

x_train = np.asarray(x_train)

y_train = list(train['BroadClassification'])

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(y_train)



def encode(le, labels):

    enc = le.transform(labels)

    return keras.utils.to_categorical(enc)



y_enc = encode(le, y_train)

y_train = np.asarray(y_enc)
print(y_train.shape)
model = build_model(bert_layer, max_len=512)

model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(

    train_input, y_train,

    validation_split=0.2,

    epochs=12,

    callbacks=[checkpoint],

    batch_size=16

)
submission = pd.read_csv("/kaggle/input/vsoactive/VSOActive.csv")

submission.head()
import pandas as pd

import numpy as np

import nltk

from nltk.corpus import stopwords

import gensim

from gensim.models import LdaModel

from gensim import models, corpora, similarities

import re

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

import time

from nltk import FreqDist

from scipy.stats import entropy

import matplotlib.pyplot as plt

import seaborn as sns

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('wordnet')

def initial_clean(text):

    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)

    text = re.sub("[^a-zA-Z ]", "", text)

    text = text.lower() # lower case the text

    text = nltk.word_tokenize(text)

    return text



stop_words = stopwords.words('english')

def remove_stop_words(text):

    return [word for word in text if word not in stop_words]



stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()

def stem_words(text):

    try:

        text = [stemmer.stem(word) for word in text]

        text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words

    except IndexError: # the word "oed" broke this, so needed try except

        pass

    return text



def apply_all(text):

    ans=""

    for x in stem_words(remove_stop_words(initial_clean(text))):

      ans = ans + x +' '

    return ans
model.load_weights('./model.h5')

submission['merged']=submission['Title']+" "+submission['Description']

submission['merged'].fillna(" ",inplace=True)

submission['merged_clean']=submission['merged'].apply(apply_all)

train_input = pre_Process_data(submission.merged_clean.values, tokenizer, max_len=512)

test_pred = model.predict(train_input)

def decode(le, one_hot):

    dec = np.argmax(one_hot, axis=1)

    return le.inverse_transform(dec)

pred=decode(le, test_pred)
p=submission['BroadClassification'].values

count=0

for i in range(len(p)):

    if(p[i]==pred[i]):

        count=count+1

    else:

        print(p[i],"-->",pred[i])

print(count/len(pred))
print(count)
for layers in model.layers:

    print(layers.name)
layer_name = 'tf_op_layer_strided_slice_1'

#model.load_weights('./model.h5')  

model.output_hidden_states=True

# with tf.Session() as session:

#     tf.keras.backend.get_session(session)

#     session.run(tf.global_variables_initializer())

#     session.run(tf.tables_initializer())

intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

intermediate_output8 = intermediate_layer_model.predict(train_input)
print(intermediate_output8.shape)
submission=pd.read_csv('/kaggle/input/vso-closed/VSO_Closed_subcategory.csv')

submission['merged_clean'].fillna(" ",inplace=True)

train_input = pre_Process_data(submission.merged_clean.values, tokenizer, max_len=512)
intermediate_output1 = intermediate_layer_model.predict(train_input)
print(intermediate_output1.shape)

np.save('./embeddings.npy',intermediate_output1)
def function(text):

    train_input = pre_Process_data([apply_all(text)], tokenizer, max_len=512)

    return intermediate_layer_model.predict(train_input)
print(function("dlp policy").shape)
submission['Title_clean']=submission['Title'].apply(apply_all)

submission['embeddings_title']=submission['Title_clean'].apply(function)

submission['Description_clean']=submission['Description'].apply(apply_all)

submission['embeddings_desc']=submission['Description_clean'].apply(function)
submission.to_csv('./VSO_Closed_subcategory_withembeddings.csv')
submission.head()
from IPython.display import FileLink

FileLink('./model.h5') 
!ls