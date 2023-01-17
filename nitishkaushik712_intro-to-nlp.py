import nltk
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
dir(nltk)
# Read in the raw text
rawData = open("../input/SMSSpamCollection.tsv").read()

# Print the raw data
rawData[0:500]
parsedData = rawData.replace('\t','\n').split("\n")
parsedData
labelList = parsedData[0::2]
textList = parsedData[1::2]
print(labelList[0:5])
print(textList[0:5])
import pandas as pd

fullCorpus = pd.DataFrame({
    'label': labelList[:-1],
    'body_list': textList
})

data = fullCorpus
# How many spam/ham are there?

print("Out of {} rows, {} are spam, {} are ham".format(len(fullCorpus),
                                                       len(fullCorpus[fullCorpus['label']=='spam']),
                                                       len(fullCorpus[fullCorpus['label']=='ham'])))
import re

re_test = 'This is a made up string to test 2 different regex methods'
re_test_messy = 'This      is a made up     string to test 2    different regex methods'
re_test_messy1 = 'This-is-a-made/up.string*to>>>>test----2""""""different~regex-methods'
re.split('\s+',re_test)
re.split('\W+',re_test_messy1)
re.findall('\w+',re_test_messy1)
pep8_test = 'I try to follow PEP8 guidelines'
pep7_test = 'I try to follow PEP7 guidelines'
peep8_test = 'I try to follow PEEP8 guidelines'
import re

re.findall('[a-z]+', pep8_test)
re.findall('[A-Z]+', pep8_test)
re.findall('[A-Z]+[0-9]+', peep8_test)
re.sub('[A-Z]+[0-9]+', 'PEP8 Python Styleguide', peep8_test)
import pandas as pd
pd.set_option('display.max_colwidth', 100)

data = pd.read_csv("../input/SMSSpamCollection.tsv", sep='\t', header=None)
data.columns = ['label', 'body_text']
data
import string
string.punctuation
def remove_punct(text):
    text_no_punct = "".join([char for char in text if char not in string.punctuation])
    return text_no_punct

data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punct(x))
data
import re
def tokens(text):
    tokens = re.split('\W+',text)
    return tokens

data['Tokens'] = data['body_text_clean'].apply(lambda x: tokens(x))
data
import nltk

stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text_no_stopwords = [char for char in text if char not in stopwords]
    return text_no_stopwords

data['body_text_clean_no_stopwords'] = data['Tokens'].apply(lambda x: remove_stopwords(x))
data
ps = nltk.PorterStemmer()
dir(ps)
print(ps.stem('grows'))
print(ps.stem('growing'))
print(ps.stem('grow'))
print(ps.stem('run'))
print(ps.stem('running'))
print(ps.stem('runner'))
def stemming(text):
    text_stemmed = [ps.stem(word) for word in text]
    return text_stemmed

data['tex_stemmed'] = data['body_text_clean_no_stopwords'].apply(lambda x: stemming(x))
data
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()
print(ps.stem('meanness'))
print(ps.stem('meaning'))
print(wn.lemmatize('goose'))
print(wn.lemmatize('geese'))
def lemmatize(text):
    lemma = [wn.lemmatize(word).lower() for word in text]
    return lemma

data['text_lemma'] = data['body_text_clean_no_stopwords'].apply(lambda x: lemmatize(x))
data
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return text
data['final_clean_data'] = data['body_text'].apply(lambda x: clean_text(x))
data
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(2,2))
X_counts = count_vect.fit_transform(data['body_text'])
print(X_counts.shape)
print(count_vect.get_feature_names())
X_counts_df = pd.DataFrame(X_counts.toarray())
X_counts_df
X_counts_df.columns = count_vect.get_feature_names()
X_counts_df
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(data['body_text'])
print(X_tfidf.shape)
print(tfidf_vect.get_feature_names())
X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
X_tfidf_df.columns = tfidf_vect.get_feature_names()
X_tfidf_df
