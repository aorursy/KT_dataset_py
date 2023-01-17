import numpy as np 

import pandas as pd

import string

import re

from collections import Counter

from keras.preprocessing.text import Tokenizer

from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer

from nltk import PorterStemmer as Stemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
pd.set_option('display.max_colwidth', None)

sms = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding = 'latin-1')

sms = pd.DataFrame({'label' : sms['v1'],'text' : sms['v2']})
sms.head(n=10)
sms.groupby('label').describe()
sms = sms.drop_duplicates(subset = 'text')

sms.groupby('label').describe()
sms['length'] = sms['text'].apply(len)

sms.hist(column='length',by='label',bins=50)
def clean_doc(docs): # Test with removing lowercase, filter 1 letter word, stemming

    # Split into tokens

    tokens = docs.split()

    # Remove punctuation, in words or stand alone

    re_punc = re.compile('[%s]' % re.escape(string.punctuation))

    tokens = [re_punc.sub(' ',w) for w in tokens]

    # Remove Stopwords

    tokens = [i for i in tokens if i not in stopwords.words('english')]

    # Lowercase

    tokens = [i.lower() for i in tokens]

    # Remove non-alphabetic entries:

    tokens = [i for i in tokens if i.isalpha()]

    # filter out 1 letter words

    # tokens = [i for i in tokens if len(i)>1]

    # Stemming

    #st = Stemmer()

    #tokens = [st.stem(t) for t in tokens]

    return(tokens)
print(sms['text'][9])

print(clean_doc(sms['text'][9]))
vec = CountVectorizer(analyzer=clean_doc)

X = vec.fit_transform(sms['text'])

print('Size of vocabulary is: '+str(len(vec.get_feature_names())))

X
len(vec.get_feature_names())
pipe = Pipeline([

('bow',CountVectorizer(analyzer=clean_doc)), 

# Since v0.21, if input is filename or file, the data is first read from the file 

# and then passed to the given callable analyzer.

('classifier',MultinomialNB())

])



pipe2 = Pipeline([

('tfidf',TfidfVectorizer(analyzer=clean_doc)), 

# Since v0.21, if input is filename or file, the data is first read from the file 

# and then passed to the given callable analyzer.

('classifier',BernoulliNB())

])
X_train, X_test, y_train, y_test = train_test_split(sms['text'],sms['label'],test_size = 0.2,random_state = 10)
pipe.fit(X_train,y_train)
pred = pipe.predict(X_test)

acc = sum(pred == y_test)

precent_acc = (acc/len(y_test)) * 100

precent_acc
print(len(X_test[y_test != pred]))

print(len(X_test))
print(classification_report(y_test,pred))
wrong_pred = pd.DataFrame({'text' : X_test[pred != y_test],'Prediction': pred[pred != y_test],'True_value':y_test[pred != y_test]})

wrong_pred
pipe.predict(['WIN URGENT! Your mobile number has been awarded with a å£2000 prize GUARANTEED call 09061790121 from land line. claim 3030 valid 12hrs only 150ppm '])