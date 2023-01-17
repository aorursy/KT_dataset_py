# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import gc

import re

import nltk

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(0)

from keras.models import Model

from keras.layers import Dense, Input, Dropout, LSTM, Activation

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from collections import  Counter

plt.style.use('ggplot')

stop=set(stopwords.words('english'))

import re

from nltk.tokenize import word_tokenize

import gensim

import string

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout,SpatialDropout1D

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

np.random.seed(1)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train= pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')

train.head()
train.isnull().sum(axis=0)
test.isnull().sum(axis=0)
keyword_cnt = train.keyword.value_counts()

keyword_cnt
train_fake = train[train['target'] == 1]

keyword_cnt_fake = train_fake.keyword.value_counts()

keyword_cnt_fake
train['target_mean']=train.groupby('keyword')['target'].transform('mean')

fig=plt.figure(figsize=(10,72),dpi=100)

sns.countplot(y=train.sort_values(by='target_mean',ascending=False)['keyword'],

              hue=train.sort_values(by='target_mean',ascending=False)['target'])

plt.tick_params(axis='x',labelsize=15)

plt.tick_params(axis='y',labelsize=12)

plt.legend(loc=1)

plt.title('Keywords distribution by target')

plt.show()

train.drop(columns=['target_mean'],inplace=True)
from wordcloud import WordCloud, STOPWORDS

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()

train_df0 = train[train['target']==0]

train_df1 = train[train['target']==1]

plot_wordcloud(train_df0['text'], title="Real Tweet Word Cloud")

plot_wordcloud(train_df1['text'], title="Fake Tweet Word Cloud")
from nltk.stem.porter import PorterStemmer

n_corpus=[]

for text in tqdm(train['text']):

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)

    text = re.sub(r'[0-9]', '', text)

    text = text.lower()

    text = nltk.word_tokenize(text)

    ps = PorterStemmer()

    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]

    text = ' '.join(text)

    n_corpus.append(text)
train['text_n']=n_corpus

train.drop('text',axis=1)
n_corpus=[]

for text in tqdm(test['text']):

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)

    text = re.sub(r'[0-9]', '', text)

    text = text.lower()

    text = nltk.word_tokenize(text)

    ps = PorterStemmer()

    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]

    text = ' '.join(text)

    n_corpus.append(text)
test['text_n']=n_corpus

test.drop('text',axis=1)
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

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



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
#downloading bert layer

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(train.text.values, tokenizer, max_len=160)

test_input = bert_encode(test.text.values, tokenizer, max_len=160)

train_labels = train.target.values
model = build_model(bert_layer, max_len=160)

model.summary()
train_history = model.fit(

    train_input, train_labels,

    validation_split=0.2,

    epochs=3,

    batch_size=16

)



model.save('model.h5')
test_pred = model.predict(test_input)

submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)


