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
# for installing Tokenization

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import nltk  # for text manipulation 

import string 

import warnings 

import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt  

import re

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub



import tokenization



pd.set_option("display.max_colwidth", 200) 

warnings.filterwarnings("ignore", category=DeprecationWarning) 



%matplotlib inline
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
print(train.shape)

train.head()
print(test.shape)

test.head()
# looking into the number of real and unreal comments

train['target'].value_counts()
#Distribution of real and unreal comment

sns.barplot(train['target'].value_counts().index,train['target'].value_counts())
# distribution of length of the tweets, in terms of words, in both train and test data.

length_train = train['text'].str.len() 

length_test = test['text'].str.len() 

plt.hist(length_train, bins=20, label="train_text") 

plt.hist(length_test, bins=20, label="test_text") 

plt.legend() 

plt.show()
#Distribution of real and unreal characters in train data

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

train_len=train[train['target']==1]['text'].str.len()

ax1.hist(train_len)

ax1.set_title('Disaster tweets')

train_len=train[train['target']==0]['text'].str.len()

ax2.hist(train_len, color="orange")

ax2.set_title('Not disaster tweets')

fig.suptitle('Characters in tweets')

plt.show()
#function for removing pattern

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

    return input_txt



# remove '#' handle

train['tweet'] = np.vectorize(remove_pattern)(train['text'], "#[\w]*")

test['tweet'] = np.vectorize(remove_pattern)(test['text'], "#[\w]*") 

train.head()
#Delete everything except alphabet

train['tweet'] = train['tweet'].str.replace("[^a-zA-Z#]", " ")

test['tweet'] = test['tweet'].str.replace("[^a-zA-Z#]", " ")

train.head()
#Dropping words whose length is less than 3

train['tweet'] = train['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

test['tweet'] = test['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

train.head()
#convert all the words into lower case

train['tweet'] = train['tweet'].str.lower()

test['tweet'] = test['tweet'].str.lower()
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 

set(stopwords.words('english'))



# set of stop words

stops = set(stopwords.words('english')) 



# tokens of words  

train['tokenized_sents'] = train.apply(lambda row: nltk.word_tokenize(row['tweet']), axis=1)

test['tokenized_sents'] = test.apply(lambda row: nltk.word_tokenize(row['tweet']), axis=1)



#function to remove stop words

def remove_stops(row):

    my_list = row['tokenized_sents']

    meaningful_words = [w for w in my_list if not w in stops]

    return (meaningful_words)



#removing stop words

train['clean_tweet'] = train.apply(remove_stops, axis=1)

test['clean_tweet'] = test.apply(remove_stops, axis=1)

train.drop(["tweet","tokenized_sents"], axis = 1, inplace = True)

test.drop(["tweet","tokenized_sents"], axis = 1, inplace = True)
train.head()
#re-join the words after tokenization

def rejoin_words(row):

    my_list = row['clean_tweet']

    joined_words = ( " ".join(my_list))

    return joined_words



train['clean_tweet'] = train.apply(rejoin_words, axis=1)

test['clean_tweet'] = test.apply(rejoin_words, axis=1)

train.head()
#Visualization of all the words using word cloud

from wordcloud import WordCloud 

all_word = ' '.join([text for text in train['clean_tweet']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_word) 

plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off') 

plt.show()
#Visualization of all the words which signify real disaster

normal_words =' '.join([text for text in train['clean_tweet'][train['target'] == 1]]) 

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words) 

plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
#Visualization of all the words which signify unreal disaster

normal_words =' '.join([text for text in train['clean_tweet'][train['target'] == 0]]) 

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words) 

plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
#Dropping the unnecessary column

train.drop(["keyword","location","text"], axis = 1, inplace = True)

test.drop(["keyword","location","text"], axis = 1, inplace = True)

train.head()
from transformers import BertTokenizer



tokenizer_QA = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer_QA.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer_QA.convert_tokens_to_ids(input_sequence)

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



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
#Bert Module

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
#tokenization

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

# tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(train.clean_tweet.values, tokenizer_QA, max_len=160)

test_input = bert_encode(test.clean_tweet.values, tokenizer_QA, max_len=160)

train_labels = train.target.values
model = build_model(bert_layer, max_len=160)

model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)



train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=3,

    callbacks=[checkpoint],

    batch_size=16

)
model.load_weights('model.h5')

test_pred = model.predict(test_input)
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submission['target'] = test_pred.round().astype(int)

submission.to_csv('new_submission.csv', index=False)
submission