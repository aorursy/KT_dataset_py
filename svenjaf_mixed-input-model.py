# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np

import os

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import re

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

import spacy

from bs4 import BeautifulSoup # Library beatifulsoup4 handles html

from nltk.corpus import stopwords



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_path = "../input/nlp-getting-started/train.csv"

test_path = "../input/nlp-getting-started/test.csv"

sample_submission_path = "../input/nlp-getting-started/sample_submission.csv"
df_train = pd.read_csv(train_path)

df_test = pd.read_csv(test_path)
df_train.head()
df_train.info()
#We do not want to keep our meta data

df_train = df_train[['text','target']]

df_test = df_test[['text']]
df_train.target.value_counts()
#Length of the tweets

df_train['text_len'] = df_train['text'].apply(len)

df_train['text_len'].describe()



df_test['text_len'] = df_test['text'].apply(len)
plt.hist(df_train.text_len)

plt.show()
f, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

f.suptitle("Histogram of char length of text",fontsize=20)

sns.distplot(df_train[df_train['target']==0].text_len.values,kde=False,bins=20,hist=True,ax=axes[0],label="Bins unreal disaster",

            kde_kws={"color": "r", "lw": 2, "label": "KDE 0"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "orange"})

axes[0].legend(loc="best")

axes[0].set_ylabel("Rows Count")

sns.distplot(df_train[df_train['target']==1].text_len.values,kde=False,bins=20,hist=True,ax=axes[1],label="Bins real disaster",

            kde_kws={"color": "g", "lw": 2, "label": "KDE 1"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "red"})

axes[1].legend(loc="best")
#number of words

def word_count(txt):

    return len(txt.split())

df_train['word_count'] = df_train.text.apply(word_count)



df_test['word_count'] = df_test.text.apply(word_count)
#average number of characters per word

df_train['char_word_count']=df_train['text_len']/df_train['word_count']



df_test['char_word_count']=df_test['text_len']/df_test['word_count']
df_train.head()
df_train.groupby(['target'])['char_word_count'].mean()
def url_count(txt):

    return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',txt))
df_train['url_count'] = df_train.text.apply(url_count)



df_test['url_count'] = df_test.text.apply(url_count)
df_train['url_count'].value_counts()
#create dummy for url instead of continuouse variable

df_train['url_d'] = (df_train['url_count'] > 0).astype(int)



df_test['url_d'] = (df_test['url_count'] > 0).astype(int)
import emoji

def emoji_count(txt):

    e_txt = emoji.demojize(txt)

    return len(re.findall(':(.*?):',e_txt))
df_train['emoji_count'] = df_train.text.apply(emoji_count)



df_test['emoji_count'] = df_test.text.apply(emoji_count)
#create dummy for dummy instead of continuouse variable

df_train['emoji_d'] = (df_train['emoji_count'] > 0).astype(int)



df_test['emoji_d'] = (df_test['emoji_count'] > 0).astype(int)
def count_hashtags(text):

    gethashtags = re.findall('#\w*[a-zA-Z]\w*',text)

    return len(gethashtags)
df_train['hash_count'] = df_train.text.apply(count_hashtags)



df_test['hash_count'] = df_test.text.apply(count_hashtags)
#exclamation marks and question marks

def count_punctuations(text):

    getpunctuation = re.findall('[?!]+?',text)

    return len(getpunctuation)
df_train['punctuation_count'] = df_train.text.apply(count_punctuations)



df_test['punctuation_count'] = df_test.text.apply(count_punctuations)




f, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

f.suptitle("Histogram of Excla/Question marks",fontsize=20)

sns.distplot(df_train[df_train['target']==0].punctuation_count.values,kde=False,bins=20,hist=True,ax=axes[0],label="Bins unreal disaster",

            kde_kws={"color": "r", "lw": 2, "label": "KDE 0"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "orange"})

axes[0].legend(loc="best")

axes[0].set_ylabel("Rows Count")

sns.distplot(df_train[df_train['target']==1].punctuation_count.values,kde=False,bins=20,hist=True,ax=axes[1],label="Bins real disaster",

            kde_kws={"color": "g", "lw": 2, "label": "KDE 1"},

                           hist_kws={ "linewidth": 2,

                                     "alpha": 1, "color": "red"})

axes[1].legend(loc="best")
df_train.head()
# Lemmatize with POS Tag

def get_wordnet_pos(word):

    """Map POS tag to first character for lemmatization"""

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}



    return tag_dict.get(tag, wordnet.NOUN)
def clean_text(df):

    

    tweets = []



    lemmatizer = WordNetLemmatizer()

    

    

    for tweet in df:

        

        # remove html content

        tweet_text = BeautifulSoup(tweet).get_text()

        

        # remove non-alphabetic characters

        tweet_text = re.sub("[^a-zA-Z]"," ", tweet_text)

    

        # tokenize the sentences

        words = word_tokenize(tweet_text.lower())

  

        # filter stopwords

        words = [w for w in words if w not in stopwords.words("english")]

        

        # lemmatize each word to its lemma

        lemma_words =[lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in words]

    

        tweets.append(lemma_words)

       



    return(tweets) 
tweets = clean_text(df_train.text)



tweets_test = clean_text(df_test.text)
# Undo the tokenization and put the data into a new column in the data frame.

from nltk.tokenize.treebank import TreebankWordDetokenizer



df_train['text'] = [TreebankWordDetokenizer().detokenize(word) for word in tweets]



df_test['text'] = [TreebankWordDetokenizer().detokenize(word) for word in tweets_test]
# We conduct a simple dictionary based sentiment anlysis by using the the package Blob  

from textblob import TextBlob

# Defining a sentiment analyser function

def sentiment_analyser(text):

    return text.apply(lambda Text: pd.Series(TextBlob(Text).sentiment.polarity))



# Applying function to reviews

df_train['Polarity'] = sentiment_analyser(df_train['text'])



df_test['Polarity'] = sentiment_analyser(df_test['text'])
# This code is adapted from a kaggle post by Hsankesara (https://www.kaggle.com/hsankesara/understanding-medium)

def get_words_count(df, col):

    words_count = {}

    m = df.shape[0]

    for i in range(m):

        words = df[col].iat[i].split()

        for word in words:

            if word.lower() in words_count:

                words_count[word.lower()] += 1

            else:

                words_count[word.lower()] = 1

    return words_count
tweet_words = get_words_count(df_train[df_train.target == 1], 'text')
tweet_words_df = pd.DataFrame(list(tweet_words.items()), columns=['words', 'count'])
## List of 30 most frequent words occurred in tweet

tweet_words_df.sort_values(by='count', ascending=False).head(30)
#remove words less than 3 characters from dataframe

tweet_words_df=tweet_words_df[tweet_words_df['words'].apply(len) > 2]
#displayed in a WordCloud

from wordcloud import WordCloud

fig = plt.figure(dpi=100)

a4_dims = (6, 12)

fig, ax = plt.subplots(figsize=a4_dims)

wordcloud = WordCloud(background_color ='white', max_words=200,max_font_size=40,random_state=3).generate(str(tweet_words_df.sort_values(by='count', ascending=False)['words'].values[:15]))

plt.imshow(wordcloud)

plt.title = 'Top Word in the disaster tweets'

plt.show()
top_tweet_words = tweet_words_df.sort_values(by='count', ascending=False)['words'].values[:30]
df_train['top_tweet_count'] = df_train['text'].apply(lambda s: sum(s.count(top_tweet_words[i]) for  i in range(30)))



df_test['top_tweet_count'] = df_test['text'].apply(lambda s: sum(s.count(top_tweet_words[i]) for  i in range(30)))
from sklearn.model_selection import train_test_split

#splitting into train and test data

x_train, x_test, y_train, y_test = train_test_split(df_train, np.ravel(df_train.target), test_size=0.2, random_state=42)
x_train1=x_train[['word_count', 'char_word_count', 'url_count', 'emoji_count', 'hash_count', 'punctuation_count', 'Polarity', 'top_tweet_count']]

x_test1=x_test[['word_count', 'char_word_count', 'url_count', 'emoji_count', 'hash_count', 'punctuation_count', 'Polarity', 'top_tweet_count']]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x_train1)#mind that we use only x_train values not to leak the data to the test set

x_train1= scaler.transform(x_train1)

x_test1= scaler.transform(x_test1) 
# Build vocabulary using Keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from gensim.models import KeyedVectors

from gensim.models.keyedvectors import Word2VecKeyedVectors
NUM_WORDS = 2000



tokenizer_obj = Tokenizer(NUM_WORDS, oov_token=1)  # We fit the tokenizer to the training set articles. The test set might include

tokenizer_obj.fit_on_texts(x_train.text)  # words that are not part of the training data. The argument oov_token ensures that such new words are mapped to the specified index
# Convert training set articles to sequences of integer values

X_tr_int = tokenizer_obj.texts_to_sequences(x_train.text)
#* Determine the maximum article length in the training set

max_tweet_length = max([len(tweet) for tweet in X_tr_int])

print('The longest tweet of the training set has {} words.'.format(max_tweet_length))
X_tr_int_pad = pad_sequences(X_tr_int, max_tweet_length)
#apply to test dataset



# Encode and pad the test data

X_ts_int = tokenizer_obj.texts_to_sequences(x_test.text)  # Due to oov_token argument, new words will be mapped to 1

X_ts_int_pad = pad_sequences(X_ts_int, max_tweet_length)
# Structure of the prepared training and test data

X_tr_int_pad.shape, y_train.shape, X_ts_int_pad.shape, y_test.shape
# Load GloVe embeddings

glove_index = {}

with open('../input/glove/glove.6B.50d.txt', 'r', encoding="utf8") as f:

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        glove_index[word] = coefs



print('Found %s word vectors.' % len(glove_index))
def get_embedding_matrix(tokenizer, pretrain, vocab_size):

    '''

        Helper function to construct an embedding matrix for 

        the focal corpus based on some pre-trained embeddings.

    '''

    

    dim = 0

    if isinstance(pretrain, KeyedVectors) or isinstance(pretrain, Word2VecKeyedVectors):

        dim = pretrain.vector_size        

    elif isinstance(pretrain, dict):

        dim = next(iter(pretrain.values())).shape[0]  # get embedding of an arbitrary word

    else:

        raise Exception('{} is not supported'.format(type(pretrain)))

    

    

    # Initialize embedding matrix

    emb_mat = np.zeros((vocab_size, dim))



    # There will be some words in our corpus for which we lack a pre-trained embedding.

    # In this tutorial, we will simply use a vector of zeros for such words. We also keep

    # track of the words to do some debugging if needed

    oov_words = []

    # Below we use the tokenizer object that created our task vocabulary. This is crucial to ensure

    # that the position of a words in our embedding matrix corresponds to its index in our integer

    # encoded input data

    for word, i in tokenizer.word_index.items():  

        # try-catch together with a zero-initilaized embedding matrix achieves our rough fix for oov words

        try:

            emb_mat[i] = pretrain[word]

        except:

            oov_words.append(word)

    print('Created embedding matrix of shape {}'.format(emb_mat.shape))

    print('Encountered {} out-of-vocabulary words.'.format(len(oov_words)))

    return (emb_mat, oov_words)
# Create matrix with Glove embeddings

glove_weights, _ = get_embedding_matrix(tokenizer_obj, glove_index, NUM_WORDS)
from keras.models import Model

from keras.models import Sequential

from keras.layers import Dense, Embedding,GRU, Dropout

from keras.layers.embeddings import Embedding

from keras.initializers import Constant

from keras.layers import Bidirectional

from keras.layers import Input

from keras.layers.merge import Concatenate 
EPOCH = 5

EMBEDDING_DIM = 50

BATCH_SIZE=32

VAL_SPLIT = 0.25
#we have two inputs

input_1 = Input(shape=(max_tweet_length,))



input_2 = Input(shape=(8,))
# submodel embedding layer

embedding_layer = Embedding(NUM_WORDS, 

                         EMBEDDING_DIM,  

                         embeddings_initializer=Constant(glove_weights), 

                         input_length=max_tweet_length, 

                         trainable=False

                         ) (input_1)

GRU_Layer_1 = Bidirectional(GRU(512))(embedding_layer)
dense_layer_1 = Dense(10, activation='relu')(input_2)

dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
concat_layer = Concatenate()([GRU_Layer_1, dense_layer_2])

dense_layer_3 = Dense(10, activation='relu')(concat_layer)

output = Dense(1, activation='sigmoid')(dense_layer_3)

model = Model(inputs=[input_1, input_2], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model_story = model.fit(x=[X_tr_int_pad, x_train1], y=y_train, validation_data=([X_ts_int_pad, x_test1], y_test), batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1 )
df_test1=df_test[['word_count', 'char_word_count', 'url_count', 'emoji_count', 'hash_count', 'punctuation_count', 'Polarity', 'top_tweet_count']]
df_test1= scaler.transform(df_test1)
# Encode and pad the test prediction data

df_ts_int = tokenizer_obj.texts_to_sequences(df_test.text) 

df_ts_int_pad = pad_sequences(df_ts_int, max_tweet_length)
#Prepare submission

# make predictions on the testing data

preds = model.predict([df_ts_int_pad , df_test1])
preds
preds=np.round(preds).astype(int)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = preds
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)