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
#the basics
import os, re, math, string, pandas as pd, numpy as np, seaborn as sns

#graphing
import matplotlib.pyplot as plt

#deep learning
import tensorflow as tf

#nlp
from wordcloud import STOPWORDS

#scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#choose batch size
BATCH_SIZE = 16

#how many epochs?
EPOCHS = 2

#clean Tweets?
CLEAN_TWEETS = False

#use meta data?
USE_META = False

#add dense layer?
ADD_DENSE = True
DENSE_DIM = 64

#add dropout?
ADD_DROPOUT = True
DROPOUT = .2

#train BERT base model? 
TRAIN_BASE = True
#get data
train = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip',sep='\t')
test = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip',sep='\t')

#peek at train
train.head()
#save ID
test_id = pd.DataFrame()
test_id = test['PhraseId']
test_id = test['SentenceId']

#drop from train and test
columns = {'PhraseId', 'SentenceId'}
train = train.drop(columns = columns)
test = test.drop(columns = columns)
#BERT
from transformers import BertTokenizer
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
enc = TOKENIZER.encode("Encode me!")
dec = TOKENIZER.decode(enc)
print("Encode: " + str(enc))
print("Decode: " + str(dec))
import tensorflow as tf
from transformers import TFBertModel, BertModel
def bert_encode(data,maximum_len) :
    input_ids = []
    attention_masks = []
  

    for i in range(len(data.Phrase)):
        encoded = TOKENIZER.encode_plus(data.Phrase[i],
                                        add_special_tokens=True,
                                        max_length=maximum_len,
                                        pad_to_max_length=True,
                                        return_attention_mask=True)
      
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        
    return np.array(input_ids),np.array(attention_masks)
def build_model(model_layer, learning_rate, use_meta = USE_META, add_dense = ADD_DENSE,
               dense_dim = DENSE_DIM, add_dropout = ADD_DROPOUT, dropout = DROPOUT):
    
    #define inputs
    input_ids = tf.keras.Input(shape=(60,),dtype='int32')
    attention_masks = tf.keras.Input(shape=(60,),dtype='int32')
    #meta_input = tf.keras.Input(shape = (meta_train.shape[1], ))
    
    #insert BERT layer
    transformer_layer = model_layer([input_ids,attention_masks])
    
    #choose only last hidden-state
    output = transformer_layer[1]
    
    #add meta data
    if use_meta:
        output = tf.keras.layers.Concatenate()([output, meta_input])
    
    #add dense relu layer
    if add_dense:
        print("Training with additional dense layer...")
        output = tf.keras.layers.Dense(dense_dim,activation='relu')(output)
    
    #add dropout
    if add_dropout:
        print("Training with dropout...")
        output = tf.keras.layers.Dropout(dropout)(output)
    
    #add final node for binary classification
    output = tf.keras.layers.Dense(5,activation='softmax')(output)
    
    #assemble and compile
    if use_meta:
        print("Training with meta-data...")
        model = tf.keras.models.Model(inputs = [input_ids,attention_masks, meta_input],outputs = output)
    
    else:
        print("Training without meta-data...")
        model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)

    model.compile(tf.keras.optimizers.Adam(lr=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

#define conveient training function to visualize learning curves
def plot_learning_curves(history): 
    fig, ax = plt.subplots(1, 2, figsize = (20, 10))

    ax[0].plot(history.history['accuracy'], color='#171820')
    ax[0].plot(history.history['val_accuracy'], '#fdc029')

    ax[1].plot(history.history['loss'], color='#171820')
    ax[1].plot(history.history['val_loss'], '#fdc029')

    ax[0].legend(['train', 'validation'], loc = 'upper left')
    ax[1].legend(['train', 'validation'], loc = 'upper left')

    fig.suptitle("Model Learning Curves", fontsize=14)

    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')

    return plt.show()
#get BERT layer
bert_large = TFBertModel.from_pretrained('bert-large-uncased')
#bert_base = BertModel.from_pretrained('bert-large-uncased')          #to use with PyTorch

#select BERT tokenizer
TOKENIZER = BertTokenizer.from_pretrained("bert-large-uncased")

#get our inputs
train_input_ids,train_attention_masks = bert_encode(train,60)
test_input_ids,test_attention_masks = bert_encode(test,60)

#debugging step
print('Train length:', len(train_input_ids))
print('Test length:', len(test_input_ids))

#and build and view parameters
BERT_large = build_model(bert_large, learning_rate = 1e-5)
BERT_large.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint('large_model.h5', monitor='val_loss', save_best_only = True, save_weights_only = True)
#train BERT
if USE_META:
    history = BERT_large.fit([train_input_ids,train_attention_masks, meta_train], train.Sentiment, validation_split = .2, epochs = EPOCHS, callbacks = [checkpoint], batch_size = BATCH_SIZE)
    
else:
    history = BERT_large.fit([train_input_ids,train_attention_masks], train.Sentiment, validation_split = .2, epochs = EPOCHS, callbacks = [checkpoint], batch_size = BATCH_SIZE)
BERT_large.load_weights('large_model.h5') 
#predict with BERT
if USE_META:
    preds_large = BERT_large.predict([test_input_ids,test_attention_masks,meta_test])

else:
    preds_large = BERT_large.predict([test_input_ids,test_attention_masks]) 
ls=[]
for i in preds_large:
    ls.append(np.argmax(i))
#save as dataframe
test = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip',sep='\t')
submission_large = test.copy()
submission_large['Sentiment'] = ls

submission_large.head(10)
submission_large = submission_large[['PhraseId', 'Sentiment']]

#save to disk
submission_large.to_csv('submission_bert_large.csv', index = False)
print('Submission saved')