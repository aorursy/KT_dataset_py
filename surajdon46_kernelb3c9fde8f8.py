# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import spacy

import re

from gensim import corpora, models, similarities

from gensim.parsing.preprocessing import preprocess_string

from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_numeric

from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, remove_stopwords, strip_short

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
np.random.seed(27)

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
from sklearn.utils import resample
sincere = train[train.target == 0]

insincere = train[train.target == 1]

train = pd.concat([resample(sincere,

                     replace = False,

                     n_samples = len(insincere)), insincere])
contractions = {

"ain't": "is not",

"aren't": "are not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he would",

"he'd've": "he would have",

"he'll": "he will",

"he'll've": "he he will have",

"he's": "he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how is",

"I'd": "I would",

"I'd've": "I would have",

"I'll": "I will",

"I'll've": "I will have",

"I'm": "I am",

"I've": "I have",

"i'd": "i would",

"i'd've": "i would have",

"i'll": "i will",

"i'll've": "i will have",

"i'm": "i am",

"i've": "i have",

"isn't": "is not",

"it'd": "it would",

"it'd've": "it would have",

"it'll": "it will",

"it'll've": "it will have",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she would",

"she'd've": "she would have",

"she'll": "she will",

"she'll've": "she will have",

"she's": "she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so as",

"that'd": "that would",

"that'd've": "that would have",

"that's": "that is",

"there'd": "there would",

"there'd've": "there would have",

"there's": "there is",

"they'd": "they would",

"they'd've": "they would have",

"they'll": "they will",

"they'll've": "they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what will",

"what'll've": "what will have",

"what're": "what are",

"what's": "what is",

"what've": "what have",

"when's": "when is",

"when've": "when have",

"where'd": "where did",

"where's": "where is",

"where've": "where have",

"who'll": "who will",

"who'll've": "who will have",

"who's": "who is",

"who've": "who have",

"why's": "why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you would",

"you'd've": "you would have",

"you'll": "you will",

"you'll've": "you will have",

"you're": "you are",

"you've": "you have"

}
contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
contractions_re
def expandContractions(text, contractions_re=contractions_re):

    def replace(match):

        return contractions[match.group(0)]

    return contractions_re.sub(replace, text)
CUSTOM_FILTERS = [lambda x: x.lower(), #lowercase

                  strip_tags, # remove html tags

                  strip_punctuation, # replace punctuation with space

                  strip_multiple_whitespaces,# remove repeating whitespaces

                  strip_non_alphanum, # remove non-alphanumeric characters

                  strip_numeric, # remove numbers

                  remove_stopwords,# remove stopwords

                  strip_short # remove words less than minsize=3 characters long

                 ]
nlp = spacy.load('en')
def gensim_preprocess(docs, logging=True):

    docs = [expandContractions(doc) for doc in docs]

    docs = [preprocess_string(text, CUSTOM_FILTERS) for text in docs]

    texts_out = []

    for doc in docs:

    # https://spacy.io/usage/processing-pipelines

        doc = nlp((" ".join(doc)),  # doc = text to tokenize => creates doc

                  # disable parts of the language processing pipeline we don't need here to speed up processing

                  disable=['ner', # named entity recognition

                           'tagger', # part-of-speech tagger

                           'textcat', # document label categorizer

                          ])

        texts_out.append([tok.lemma_ for tok in doc if tok.lemma_ != '-PRON-'])

    return pd.Series(texts_out)
gensim_preprocess(train.question_text.iloc[10:15])
train_data = gensim_preprocess(train.question_text)
train_data
ngram_phraser = models.Phrases(train_data, threshold=1)
ngram = models.phrases.Phraser(ngram_phraser)
print(ngram[train_data[0]])
texts = [ngram[token] for token in train_data]
texts
texts = [' '.join(text) for text in texts]
texts
train['ngrams'] = texts
train.head()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(train.ngrams)
x_train, x_test, y_train, y_test = train_test_split(train.ngrams, train.target, test_size=0.2)
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
%time bnb.fit(vectorizer.transform(x_train), y_train)
print('Naive Bayes Score: ', bnb.score(vectorizer.transform(x_test), y_test))
predictions = bnb.predict(vectorizer.transform(x_test))
from sklearn import metrics
print("Classification Report")

print(metrics.classification_report(y_test, predictions))

print("")

print("Confusion Matrix")

print(metrics.confusion_matrix(y_test, predictions))

print("")
predict_test = bnb.predict(vectorizer.transform(test.question_text))
predict_test
submission = pd.DataFrame({'qid':test['qid'],'prediction':predict_test})
submission.to_csv('submission.csv',index=False)