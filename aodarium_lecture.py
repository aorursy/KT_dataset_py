import numpy as np # linear algebra

import pandas as pd # data processing
import gzip

import gensim 

import logging

import requests
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#data_url = 'https://www.gutenberg.org/files/1703/1703-0.txt'

#text = requests.get(data_url).text.split('.')
!ls ../input/

data = pd.read_csv('../input/nips-2015-papers/Papers.csv')
#Create a function which return lines preprocessed thanks to gensim.utils.simple_preprocess



def read_line(file):

  ratio = len(file)/100

  for i,line in enumerate(file):

    if (i % ratio == 0): logging.info("read {0} lines".format(i))

    yield gensim.utils.simple_preprocess(line)
documents = list (read_line(data['PaperText']))

print(documents[1])
model = gensim.models.Word2Vec(

        documents,

        size=200,

        window=10,

        min_count=2,

        workers=10)

model.train(documents, total_examples=len(documents), epochs=10)



w1 = "development"

model.wv.most_similar (positive=w1)
w1 = "man"

model.wv.most_similar (positive=w1)
w1 = "woman"

model.wv.most_similar (positive=w1)
w1 = "dog"

model.wv.most_similar (positive=w1)
w1 = "god"

model.wv.most_similar (positive=w1)
data = pd.read_csv('../input/twitter-emotion/train.csv', engine='python')
data.head()
#No need of the first row

data = data[['Sentiment', 'SentimentText']]
data.head()
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")
def tokenize(tweet):

    try:

        tweet = str(tweet.lower())

        tokens = tokenizer.tokenize(tweet)

        tokens = [tok for tok in tokens if '#' not in tok]

        tokens = [tok for tok in tokens if 'http' not in tok]

        tokens = [tok for tok in tokens if '@' not in tok]

        return tokens

    except:

        return 'NC'
def postprocess(data, n=1000000):

    data = data.head(n)

    data['tokens'] = data['SentimentText'].progress_map(tokenize)

    data = data[data.tokens != 'NC']

    data.reset_index(inplace=True)

    data.drop('index', inplace=True, axis=1)

    return data



data = postprocess(data)
data.head().tokens
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(100000).tokens),

                                                    np.array(data.head(100000).Sentiment), test_size=0.2)

x_train
import gensim

from gensim.models.word2vec import Word2Vec

LabeledSentence = gensim.models.doc2vec.LabeledSentence



def labelizeTweets(tweets, label_type):

    labelized = []

    for i,v in tqdm(enumerate(tweets)):

        label = '%s_%s'%(label_type,i)

        labelized.append(LabeledSentence(v, [label]))

    return labelized



x_train = labelizeTweets(x_train, 'TRAIN')

x_test = labelizeTweets(x_test, 'TEST')
x_train[10]
model = Word2Vec(size=200, min_count=5)

model.build_vocab([x.words for x in tqdm(x_train)])

model.train([x.words for x in tqdm(x_train)], total_examples=len([x.words for x in tqdm(x_train)]),epochs=10)
model.most_similar(['vodka'])

#model.most_similar(positive=['vodka', 'france'], negative=['russia'])

#Paris

w1 = "man"

model.wv.most_similar (positive=w1)

w1 = "woman"

model.wv.most_similar (positive=w1)

w1 = "dog"

model.wv.most_similar (positive=w1)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)

matrix = vectorizer.fit_transform([x.words for x in x_train])

tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

print ('vocab size :', len(tfidf))
def buildWordVector(tokens, size):

    vec = np.zeros(size).reshape((1, size))

    count = 0.

    for word in tokens:

        try:

            vec += model[word].reshape((1, size)) * tfidf[word]

            count += 1.

        except KeyError: # handling the case where the token is not

                         # in the corpus. useful for testing.

            continue

    if count != 0:

        vec /= count

    return vec
from sklearn.preprocessing import scale

train_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, x_train))])

train_vecs_w2v = scale(train_vecs_w2v)



test_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, x_test))])

test_vecs_w2v = scale(test_vecs_w2v)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation



modelF = Sequential()

modelF.add(Dense(32, activation='relu', input_dim=200))

modelF.add(Dense(1, activation='sigmoid'))

modelF.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])



modelF.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)
score = modelF.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)

print (score[1])

print(x_test[1], y_test[1])


