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

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from nltk.classify import SklearnClassifier

from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 

stop_words_set = set(stopwords.words('english'))
from wordcloud import WordCloud,STOPWORDS

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print(train.shape)

print(train.columns)
train['text'] = train['text'].str.replace("[^A-Za-z ]", "")
train_new , test_new = train_test_split(train,test_size=0.3)
train_1 = train_new[ train_new['target'] == 1]

train_1 = train_1['text']

train_0 = train_new[ train_new['target'] == 0]

train_0 = train_0['text']
def wordcloud_draw(data, color='black'):

    words= ' '.join(data)

    cleaned_word = " ".join([word for word in words.split()

                            if 'http' not in word

#                                 and not word.startswith('@')

#                                 and not word.startswith('#')

#                                 and word.find("û") == -1

                                and word != 'RT'

                                and word != 'rt'

                            ])

    wordcloud = WordCloud(stopwords=STOPWORDS, 

                          background_color=color,

                          width=2500,

                          height=2000,

                          collocations=False

                          ).generate(cleaned_word)

    plt.figure(1,figsize=(13, 13))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
print("Simple words")

wordcloud_draw(train_0,'white')
print("disasterous words")

wordcloud_draw(train_1)
import nltk

from nltk.stem.porter import *

from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
tweets = []

stopwords_set = set(stopwords.words("english"))



for index, row in train.iterrows():

    words_filtered = [e.lower() for e in row.text.split() if len(e) > 3]

    words_cleaned = [word for word in words_filtered

        if 'http' not in word

        and not word.startswith('@')

        and not word.startswith('#')

        and word.find("û") == -1

        and word != 'rt']

    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]

#     words_with_lemmatization = [wordnet_lemmatizer.lemmatize(word) for word in words_without_stopwords ]

#     words_with_stemming = [stemmer.stem(word) for word in words_without_stopwords ]

    #words_without_stopwords.apply(lambda x: [stemmer.stem(i) for i in x])

    tweets.append((words_without_stopwords, row.target))
def get_word_features(tweets):

    all = []

    for (words, sentiment) in tweets:

        all.extend(words)

    wordlist = nltk.FreqDist(all)

    print(wordlist)

    features = wordlist.keys()

    return features
w_features = get_word_features(tweets)
def extract_features(document):

    document_words = set(document)

    features = {}

    for word in w_features:

        features['contains(%s)' % word] = (word in document_words)

    return features

wordcloud_draw(w_features)
training_set = nltk.classify.apply_features(extract_features,tweets)

classifier = nltk.NaiveBayesClassifier.train(training_set)
test_0 = test_new[test_new['target'] == 0]

test_0 = test_0['text']

test_1 = test_new[test_new['target'] == 1]

test_1 = test_1['text']
neg_cnt = 0

pos_cnt = 0

for obj in test_0: 

    

    res =  classifier.classify(extract_features(obj.split()))

   

    if(res == 0): 

        pos_cnt = pos_cnt + 1

for obj in test_1: 

    res =  classifier.classify(extract_features(obj.split()))

    

    if(res == 1): 

        neg_cnt = neg_cnt + 1
print('[Negative]: %s/%s '  % (len(test_0),pos_cnt))        

print('[Positive]: %s/%s '  % (len(test_1),neg_cnt)) 