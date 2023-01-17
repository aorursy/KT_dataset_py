import pandas as pd

import nltk, re, string, collections

from nltk.util import ngrams # function for making ngrams

import re

from nltk.corpus import stopwords

nltk.download('stopwords')

from nltk.tokenize import word_tokenize
df = pd.read_csv('/kaggle/input/coronawhy/dataset_v6.csv')
df['text'] = df['text'].astype(str)

df['text'] = df['text'].str.lower()
filter_keywords = ['age']

df = df[df['text'].str.contains('|'.join(filter_keywords))]
text_to_search = ' '.join(df["text"])
del df
# get rid of punctuation

punctuationNoPeriod = "[" + re.sub("\.","",string.punctuation) + "]"

text_to_search = re.sub(punctuationNoPeriod, "", text_to_search)
# let's remove stop words

# we will use the stop words provided by the NLTK

# we will also add in some customized stop words used in other places for COVID-19



customized_stop_words = [

    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 

    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'fig', 'fig.', 'al.', 'q', 'license',

    'di', 'la', 'il', 'del', 'le', 'della', 'dei', 'delle', 'una', 'da',  'dell',  'non', 'si', 'holder',

    'p', 'h'

]



stop_words = list(stopwords.words('english')) + customized_stop_words

print(stop_words)
# let's tokenize the words

text_tokens = word_tokenize(text_to_search)

text_to_search = [word for word in text_tokens if not word in stop_words]
# and get a list of all the bigrams

esBigrams = ngrams(text_to_search, 2)



# get the frequency of each bigram in our corpus

esBigramFreq = collections.Counter(esBigrams)



# what are the ten most popular bigrams

esBigramFreq.most_common(25)
# and get a list of all the trigrams

esTrigrams = ngrams(text_to_search, 3)



# get the frequency of each trigram in our corpus

esTrigramFreq = collections.Counter(esTrigrams)



# what are the ten most popular trigrams

esTrigramFreq.most_common(25)
del esBigrams

del esBigramFreq

del esTrigrams

del esTrigramFreq
search_for_word = 'age' # Text we want the Bi/Trigrams to contain



# reset the Bigrams

esBigrams = ngrams(text_to_search, 2)

esBigramFreq = collections.Counter(esBigrams)
for gram, freq in esBigramFreq.most_common():

    if gram[0] == search_for_word or gram[1] == search_for_word:

        print(gram, freq)
del esBigrams

del esBigramFreq
# reset the Trigrams

esTrigrams = ngrams(text_to_search, 3)

esTrigramFreq = collections.Counter(esTrigrams)
for gram, freq in esTrigramFreq.most_common():

    if gram[0] == search_for_word or gram[1] == search_for_word or gram[2] == search_for_word:

        print(gram, freq)