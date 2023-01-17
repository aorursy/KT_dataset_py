
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


def analyze_tweet(tweet):
    result = {}
    result['MENTIONS'] = tweet.count('USER_MENTION')
    result['URLS'] = tweet.count('URL')
    result['POS_EMOS'] = tweet.count('EMO_POS')
    result['NEG_EMOS'] = tweet.count('EMO_NEG')
    tweet = tweet.replace('USER_MENTION', '').replace(
        'URL', '')
    words = tweet.split()
    result['WORDS'] = len(words)
    bigrams = get_bigrams(words)
    result['BIGRAMS'] = len(bigrams)
    return result, words, bigrams


def get_bigrams(tweet_words):
    bigrams = []
    num_words = len(tweet_words)
    for i in range(num_words - 1):
        bigrams.append((tweet_words[i], tweet_words[i + 1]))
    return bigrams


def get_bigram_freqdist(bigrams):
    freq_dict = {}
    for bigram in bigrams:
        if freq_dict.get(bigram):
            freq_dict[bigram] += 1
        else:
            freq_dict[bigram] = 1
    counter = Counter(freq_dict)
    return counter

import re
import sys
from nltk.stem.porter import PorterStemmer


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            if use_stemmer:
                word = str(porter_stemmer.stem(word))
            processed_tweet.append(word)

    return ' '.join(processed_tweet)


def preprocess_csv(csv_file_name, processed_file_name, test_file=False):
    save_to_file = open(processed_file_name, 'w')

    with open(csv_file_name, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            tweet_id = line[:line.find(',')]
            if not test_file:
                line = line[1 + line.find(','):]
                positive = int(line[:line.find(',')])
            line = line[1 + line.find(','):]
            tweet = line
            processed_tweet = preprocess_tweet(tweet)
            if not test_file:
                save_to_file.write('%s,%d,%s\n' %
                                   (tweet_id, positive, processed_tweet))
            else:
                save_to_file.write('%s,%s\n' %
                                   (tweet_id, processed_tweet))
            
    save_to_file.close()
    print( '\nSaved processed tweets to: %s' % processed_file_name)
    return processed_file_name


if __name__ == '__main__':
   
    use_stemmer = False
    csv_file_name = '../input/dataset2.csv'
    processed_file_name = 'data_processed.csv'
    preprocess_csv(csv_file_name, processed_file_name, test_file=True)

if __name__ == '__main__':
   
    use_stemmer = False
    csv_file_name = '../input/dataset2_interim_eval.csv'
    processed_file_name = 'data_processed_test.csv'
    preprocess_csv(csv_file_name, processed_file_name, test_file=True)


train_df = pd.read_csv('data_processed.csv')
test_df = pd.read_csv('data_processed_test.csv')
train_df['sentiment'] = pd.read_csv('../input/dataset2.csv')['sentiment']
train_df.columns=['a','text','sentiment']
test_df.columns=['a','text']
train_df.head()
def feat(data):
    ment=[]
    url=[]
    pe=[]
    ne=[]
    bg=[]
    for i in range(len(data)):
        ment.append(analyze_tweet(data['text'].values[i])[0]['MENTIONS'])
        url.append(analyze_tweet(data['text'].values[i])[0]['URLS'])
        pe.append(analyze_tweet(data['text'].values[i])[0]['POS_EMOS'])
        ne.append(analyze_tweet(data['text'].values[i])[0]['NEG_EMOS'])
        bg.append(analyze_tweet(data['text'].values[i])[0]['BIGRAMS'])
    data['ment']=ment
    data['url']=url
    data['pe']=pe
    data['ne']=ne
    data['bg']=bg
    return data
import pandas as pd

import numpy as np

import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,GRU
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

embed_size = 300 
max_features = 50000 
maxlen = 100 



train_df=feat(train_df)
test_df=feat(test_df)

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_df.text))
train_X = tokenizer.texts_to_sequences(train_df.text)
test_X = tokenizer.texts_to_sequences(test_df.text)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)
train_X = pd.concat([pd.DataFrame(train_X),train_df[['ment','pe','ne','url','bg']].reset_index(drop=True)],axis=1)
test_X = pd.concat([pd.DataFrame(test_X),train_df[['ment','pe','ne','url','bg']].reset_index(drop=True)],axis=1)
## Get the sentiment values
train_y = train_df['sentiment'].values

train_x,val_X,train_Y, val_Y = train_test_split(train_X,train_y
                                    , test_size=0.1, random_state=2018)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = xgb.XGBClassifier()

clf.fit(train_x,train_Y)

from sklearn.metrics import accuracy_score
print(accuracy_score(val_Y,clf.predict(val_X)))


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_df.text))
train_X = tokenizer.texts_to_sequences(train_df.text)
test_X = tokenizer.texts_to_sequences(test_df.text)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)
train_X = pd.concat([pd.DataFrame(train_X),train_df[['ment','pe','ne','url','bg']].reset_index(drop=True)],axis=1)
test_X = pd.concat([pd.DataFrame(test_X),train_df[['ment','pe','ne','url','bg']].reset_index(drop=True)],axis=1)
## Get the sentiment values
train_y = train_df['sentiment'].values

train_x,val_X,train_Y, val_Y = train_test_split(train_X,train_y
                                    , test_size=0.1, random_state=2018)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = xgb.XGBClassifier()
clf.fit(train_x,train_Y)

from sklearn.metrics import accuracy_score
print(accuracy_score(val_Y,clf.predict(val_X)))



train_df.head()

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

embed_size = 300 
max_features = 50000 
maxlen = 100 

## fill up the missing values
train_X = train_df["text"].fillna("_na_").values
val_X = val_df["text"].fillna("_na_").values
test_X = test_df["text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the sentiment values
train_y = train_df['sentiment'].values
val_y = val_df['sentiment'].values

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(train_X,train_y)

from sklearn.metrics import accuracy_score
print(accuracy_score(val_y,clf.predict(val_X)))


def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()

model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))



