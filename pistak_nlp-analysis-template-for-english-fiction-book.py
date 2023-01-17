# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string # special operations on strings

import spacy # language models



from matplotlib.pyplot import imread

from matplotlib import pyplot as plt

from wordcloud import WordCloud

%matplotlib inline
!python -m spacy download en_core_web_md
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

filename = '/kaggle/input/cthulhu/The-Call-of-Cthulhu.txt'

with open(filename) as f:

    book = f.readlines()

# you may also want to remove whitespace characters like `\n` at the end of each line

book[0:10]
len(book)
book = [x.strip() for x in book] # removes line breaks

book = [x for x in book if x] # removes empty strings, because they are considered in Python as False

book[0:10]
# we see that we don't need first four lines

core_book = book[4:]

core_book[0:10]
# Joining the list into one string/text

text = ' '.join(core_book)

len(text)
no_punc_text = text.translate(str.maketrans('', '', string.punctuation))

no_punc_text[0:550]
len(text) - len(no_punc_text)
from nltk.corpus import stopwords

print(stopwords.words('english')[0:25])
from nltk.tokenize import word_tokenize

text_tokens = word_tokenize(no_punc_text)

print(text_tokens[0:50])
len(text_tokens)
my_stop_words = stopwords.words('english')

my_stop_words.append('the')

no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]

print(no_stop_tokens[0:40])
len(no_stop_tokens)
lower_words = [x.lower() for x in no_stop_tokens]

print(lower_words[0:25])
from nltk.stem import PorterStemmer

ps = PorterStemmer()

stemmed_tokens = [ps.stem(word) for word in lower_words]

print(stemmed_tokens[0:40])
# NLP english language model of spacy library

nlp = spacy.load('en')
# convert text into words with language properties, lemmas being one of them, but mostly POS, which will follow later

doc = nlp(' '.join(no_stop_tokens))

print(doc[0:40])
lemmas = [token.lemma_ for token in doc]

print(lemmas[0:25])
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(lemmas)

X
print(vectorizer.get_feature_names()[40:90])
print(X.toarray())
sum_words = X.sum(axis=0)

words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

words_freq[0:25]
# Look this up yourself and fill in the code :) This was not part of theory, but it's a bonus task for special reasearch
one_block = book[94]

doc_block = nlp(one_block)

spacy.displacy.render(doc_block, style='ent', jupyter=True)
for token in doc_block[0:20]:

    print(token, token.pos_)
nouns_verbs = [token.text for token in doc if token.pos_ in ('NOUN', 'VERB')]

print(nouns_verbs[5:25])
cv = CountVectorizer()



X = vectorizer.fit_transform(nouns_verbs)

sum_words = X.sum(axis=0)

words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

wf_df = pd.DataFrame(words_freq)

wf_df.columns = ['word', 'count']

wf_df[0:10]
afinn = pd.read_csv('/kaggle/input/bing-nrc-afinn-lexicons/Afinn.csv', sep=',', encoding='latin-1')

afinn.shape
afinn.head()
from itertools import islice



def take(n, iterable):

    "Return first n items of the iterable as a list"

    return list(islice(iterable, n))
affinity_scores = afinn.set_index('word')['value'].to_dict()

take(20, affinity_scores.items())
from nltk import tokenize

sentences = tokenize.sent_tokenize(" ".join(core_book))

sentences[5:15]
sent_df = pd.DataFrame(sentences, columns=['sentence'])

sent_df
nlp = spacy.load('en')

sentiment_lexicon = affinity_scores



def calculate_sentiment(text: str = None) -> float:

    sent_score = 0

    if text:

        sentence = nlp(text)

        for word in sentence:

            sent_score += sentiment_lexicon.get(word.lemma_, 0)

    return sent_score
# test that it works

calculate_sentiment(text = 'Amazing boys, very good!')
sent_df['sentiment_value'] = sent_df['sentence'].apply(calculate_sentiment)
# how many words are in the sentence?

sent_df['word_count'] = sent_df['sentence'].str.split().apply(len)

sent_df['word_count'].head(10)
sent_df.sort_values(by='sentiment_value').tail(10)
# Sentiment score of the whole book

sent_df['sentiment_value'].sum()
wf_df[0:10].plot.bar(x='word', figsize=(12,8), title='Top verbs and nouns')
wordcloud = WordCloud(background_color ='black', 

                       min_font_size = 10).generate(text)

plt.figure(figsize = (12, 10), facecolor = None) 

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
sent_df.plot.scatter(x='word_count', y='sentiment_value', figsize=(12,8), title='Sentence sentiment value to sentence word count')
from scipy.stats import pearsonr

corr, _ = pearsonr(sent_df['word_count'], sent_df['sentiment_value'])

corr