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
f = open('../input/lakshmied/lakshmiED.txt','r',encoding='utf-8')

text = f.read()

text = text.lower()
import string

translate_table = dict((ord(char), None) for char in string.punctuation)   

text = text.translate(translate_table)
from nltk.tokenize import RegexpTokenizer



tokenizer = RegexpTokenizer(r'\w+')

tokenizer.tokenize(text)

print(text)
#sentence tokenization

from nltk import sent_tokenize, word_tokenize

sent_tokenize(text)
#word tokenization

for sent in sent_tokenize(text):

    print(word_tokenize(sent))
#removing stop-words

from nltk.corpus import stopwords



stopwords_en = stopwords.words('english')

EN_Stopwords = set(stopwords.words('english')) # Set checking is faster in Python than list.

# Tokenize and lowercase

tokenized_lowercase = list(map(str.lower, word_tokenize(text)))

stopwords_english = set(stopwords.words('english')) # Set checking is faster in Python than list.

print([word for word in tokenized_lowercase if word not in stopwords_en])
#define punchuation

from string import punctuation

print('From string.punctuation:', type(punctuation), punctuation)

punct_stopwords = stopwords_english.union(set(punctuation))
punch_stop_word= [word for word in tokenized_lowercase if word not in punct_stopwords]

print(punch_stop_word)
#stemming

from nltk.stem import PorterStemmer



porter = PorterStemmer()



for word in punch_stop_word:

    print(porter.stem(word))
#lemmatization

from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

for word in punch_stop_word:

    print(wnl.lemmatize(word))
wnl = WordNetLemmatizer()



def penn2morphy(penntag):

    """ Converts Penn Treebank tags to WordNet. """

    morphy_tag = {'NN':'n', 'JJ':'a',

                  'VB':'v', 'RB':'r'}

    try:

        return morphy_tag[penntag[:2]]

    except:

        return 'n' # if mapping isn't found, fall back to Noun.
#POS tagging

from nltk import pos_tag

def lemmatize_sent(text): 

    # Text input is string, returns lowercased strings.

    return[wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 

            for word, tag in pos_tag(word_tokenize(text))]
print('Raw Text Before Lemmatization ')

print(text, '\n')

print('Raw Text After Stop word Removal & Lemmaztization \n')

print([word for word in lemmatize_sent(text) 

       if word not in stopwords_english

       and not word.isdigit() ])
from __future__ import print_function

import collections

import math

import numpy as np

import os

import random

import tensorflow as tf

from matplotlib import pylab

from six.moves import range

from six.moves.urllib.request import urlretrieve

from sklearn.manifold import TSNE
vocabulary_size = 50000

f = open('../input/lakshmied/lakshmiED.txt','r',encoding='utf-8')

text_cbow = f.read()

text_cbow = text.lower()

print(text_cbow)

def build_dataset(text_cbow):

    count = [['UNK', -1]]

    count.extend(collections.Counter(text_cbow).most_common(vocabulary_size - 1))

    dictionary= dict()

    for word, _ in count:

        print(word)

        dictionary[word] = len(dictionary)

    data = list()

    unk_count = 0

    for word in text_cbow:

        if word in dictionary:

            index = dictionary[word]

        else:

            index = 0  # dictionary['UNK']

            unk_count = unk_count + 1

        data.append(index)

    count[0][1] = unk_count

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 

    return data, count, dictionary, reverse_dictionary



data, count, dictionary, reverse_dictionary = build_dataset(text_cbow)

print('Most common words (+UNK)', count[:10])

print('Sample data', data[:10])

#del words  # Hint to reduce memory.
from keras.preprocessing import text

from keras.utils import np_utils

from keras.preprocessing import sequence



f = open('../input/lakshmied/lakshmiED.txt','r',encoding='utf-8')

text_cbow = f.read()



tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(text_cbow.split())

word2id = tokenizer.word_index



# build vocabulary of unique words

word2id['PAD'] = 0

id2word = {v:k for k, v in word2id.items()}

wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in text_cbow.split()]



vocab_size = len(word2id)

embed_size = 100

window_size = 2 # context window size



print('Vocabulary Size:', vocab_size)

print('Vocabulary Sample:', list(word2id.items())[:20])
def generate_context_word_pairs(corpus, window_size, vocab_size):

    context_length = window_size*2

    for words in corpus:

        sentence_length = len(words)

        for index, word in enumerate(words):

            context_words = []

            label_word   = []            

            start = index - window_size

            end = index + window_size + 1

            

            context_words.append([words[i] 

                                 for i in range(start, end) 

                                 if 0 <= i < sentence_length 

                                 and i != index])

            label_word.append(word)



            x = sequence.pad_sequences(context_words, maxlen=context_length)

            y = np_utils.to_categorical(label_word, vocab_size)

            yield (x, y)

            

            

# Test this out for some samples

i = 0

for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):

    if 0 not in x[0]:

        print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[np.argwhere(y[0])[0][0]])

    

        if i == 10:

            break

        i += 1
import keras.backend as K

from keras.models import Sequential

from keras.layers import Dense, Embedding, Lambda

import pydot



# build CBOW architecture

cbow = Sequential()

cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))

cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))

cbow.add(Dense(vocab_size, activation='softmax'))

cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')



# view model summary

print(cbow.summary())
for epoch in range(1, 10):

    loss = 0.

    i = 0

    for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):

        i += 1

        loss += cbow.train_on_batch(x, y)

        if i % 1000 == 0:

            print('Processed {} (context, word) pairs'.format(i))



    print('Epoch:', epoch, '\tLoss:', loss)

    print()
import pandas as pd

weights = cbow.get_weights()[0]

weights = weights[1:]

print(weights.shape)



pd.DataFrame(weights, index=list(id2word.values())[1:]).head()
from sklearn.metrics.pairwise import euclidean_distances



# compute pairwise distance matrix

distance_matrix = euclidean_distances(weights)

print(distance_matrix.shape)



# view contextually similar words

similar_words = {search_term: [id2word[idx] for idx in distance_matrix[word2id[search_term]-1].argsort()[1:6]+1] 

                   for search_term in ['place','travel','simple']}



similar_words
import pandas as pd 

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer 

data = [text,'between a sweep of mountains and an expanse of dark waters a 14story building looms over prince william sound most of whittier alaska’s 280 residents live in the peachcolored confines of begich tower which was built in 1956 as a us army barracks the building has its own post office and grocery store an underground tunnel leads to the town’s small school “we are our own petri dish—we share the same ventilation system” says jim hunt the city’s manager\n\nwhen covid19 reached the state controlling visitors was the town’s best hope of keeping the disease at bay—and for months they did there are only two ways to reach whittier by boat or driving the 60 miles from anchorage passing through a singlelane tunnel then in june 11 seasonal seafood processors tested positive for covid19 and departed to isolate in anchorage a month later two more cases appeared among workers at businesses along the harbor finally in august the virus penetrated begich tower an employee who worked in maintenance—which includes covid19 disinfection—tested positive along with five members of his family\n\nthe employee chose to get tested in anchorage and there’s no obligation between the two city governments to discuss cases but the busybody nature common to small towns eventually delivered the information to the city manager without the rumor mill hunt might not have known\n\n“if you test positive in another community we wouldn’t know” says hunt adding bluntly that “we have no contact tracing—none—outside of anecdotal evidence you need human resources for contact tracing”\n\nwhittier’s dilemma may sound extreme but it’s become an alarmingly common problem contact tracing the process of identifying who may have come into contact with a known case of covid19 requires people training funding in atrisk places and time—all resources the united states has been short of devoting during the pandemic'] 

tfidf_vectorizer =TfidfVectorizer(stop_words='english')

tfidf_feature = tfidf_vectorizer.fit_transform(data) 
data_frame=pd.DataFrame(data = tfidf_feature.todense(), columns=tfidf_vectorizer.get_feature_names()) 

data_frame 
