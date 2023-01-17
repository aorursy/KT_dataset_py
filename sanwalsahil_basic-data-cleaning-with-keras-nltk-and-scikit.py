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

import nltk
trainData = pd.read_csv('../input/nlp-getting-started/train.csv')
trainData.head()
prac = trainData[['text']][:5]

prac
pList = prac['text'].tolist()

pList
from nltk.tokenize import sent_tokenize
demoText = ''

for text in pList:

    demoText = demoText+text

demoText
sent_tokenize(demoText)[0]
sent_tokenize(pList[0])
from nltk.tokenize import word_tokenize
word_tokenize(pList[0])
word_tokenize(demoText)
from nltk.corpus import stopwords
# list of stop words in english language

stopwords.words('english')
demoText
len(demoText)
#breaking sentence into words

textTok = word_tokenize(demoText)

textTok
len(textTok)
# removing stopwords

texTokNew = [w for w in textTok if w not in stopwords.words('english')]

texTokNew
len(texTokNew)
import re

import string
# prepare regex for char filtering

re_punc = re.compile('[%s]' % re.escape(string.punctuation))

re_punc
# remove punctuation from each word

stripped = [re_punc.sub('', w) for w in textTok]

print(stripped[:20])

print(textTok[:20])

len(stripped)


# remove remaining tokens that are not alphabetic

words = [word for word in stripped if word.isalpha()]

print(words)

len(words)
from nltk.stem import PorterStemmer
len(words)
ps = PorterStemmer()

stemmedWords = [ps.stem(word) for word in words]

print(len(stemmedWords))

stemmedWords
for i in range(len(words)):

    print('stemmed word : '+stemmedWords[i]+ '  |||  real word: ' + words[i])
from nltk import WordNetLemmatizer

lem = WordNetLemmatizer()
for i in range(len(words)):

    print('lem word  : '+lem.lemmatize(words[i]))

    print('real word : '+words[i])

    print('stem word : '+ps.stem(words[i]))

    print('----------------------------------------------------------------')
pList
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
cv.fit(pList)
cvNew = cv.transform(pList)
cvNew
cvNew.toarray()
cv.vocabulary_
pList
cvNew.toarray()
textRandom = ['our blah scooby']

vector = cv.transform(textRandom).toarray()
vector
textRandom = ['our blah scooby our our Deeds']

vector = cv.transform(textRandom).toarray()

vector
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
pList
tf.fit(pList)
pNew = tf.transform(pList)

pNew
pNew.toarray()[0]
tf.vocabulary_
tf.inverse_transform(pNew)
from sklearn.feature_extraction.text import HashingVectorizer

hv = HashingVectorizer()
pList
hv.fit(pList)
pNew = hv.transform(pList)

pNew
pNew.toarray()[0]
pNew.shape
hv = HashingVectorizer()

pNew = hv.transform(pList)

pNew
import keras
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(demoText)
vocab_size = len(text_to_word_sequence(demoText))

vocab_size
from keras.preprocessing.text import one_hot

result = one_hot(demoText,vocab_size)

result
from keras.preprocessing.text import hashing_trick

result = hashing_trick(demoText,vocab_size,hash_function='md5')

result
from keras.preprocessing.text import Tokenizer
pList
t = Tokenizer()

fitText = t.fit_on_texts(pList)
print(t.word_counts)
pList
print(t.document_count)
print(t.word_index)
print(t.word_docs)
myList = ['hello how are are','are you']

tnew = Tokenizer()

tnew.fit_on_texts(myList)

print(tnew.word_counts)

print(tnew.word_docs)
encodedText = t.texts_to_matrix(myList,mode='binary')

encodedText
encodedText = t.texts_to_matrix(myList,mode='count')

encodedText
encodedText = t.texts_to_matrix(myList,mode='tfidf')

encodedText
encodedText = t.texts_to_matrix(myList,mode='freq')

encodedText
data = ['How are you','i am fine, how are you doing?']

data
tBOW = Tokenizer()

tBOW.fit_on_texts(data)

tBOW.word_index
encBinary = tBOW.texts_to_matrix(data,mode='binary')

encBinary
encTFIDF = tBOW.texts_to_matrix(data,mode='tfidf')

encTFIDF
encCOUNT = tBOW.texts_to_matrix(data,mode='count')

encCOUNT
encFREQ = tBOW.texts_to_matrix(data,mode='freq')

encFREQ