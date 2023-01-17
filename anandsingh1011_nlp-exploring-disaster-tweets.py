

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from matplotlib import style

style.use('ggplot')



import seaborn as sns

import string

import nltk

import warnings 

warnings.filterwarnings("ignore", category=DeprecationWarning)



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/nlp-getting-started/train.csv')
df.head()
df.info()
# Apply a first round of text cleaning techniques

import re

import string



def clean_text_round1(text):

    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



round1 = lambda x: clean_text_round1(x)
# Let's take a look at the updated text

df['clean_text'] = df.text.apply(round1)

df.head(2)
# Apply a second round of cleaning

def clean_text_round2(text):

    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''

    text = re.sub('[‘’“”…]', '', text)

    text = re.sub('\n', '', text)

    return text



round2 = lambda x: clean_text_round2(x)
# Let's take a look at the updated text

df['clean_text'] = df.text.apply(round2)

df.head(2)
# remove special characters, numbers, punctuations

df['clean_text'] = df['clean_text'].str.replace("[^a-zA-Z#]", " ")
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
tokenized_tweet = df['clean_text'].apply(lambda x: x.split())

tokenized_tweet.head()
from nltk.stem.porter import *

stemmer = PorterStemmer()



tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

tokenized_tweet.head()
for i in range(len(tokenized_tweet)):

    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])



df['clean_text'] = tokenized_tweet
all_words = ' '.join([text for text in df['clean_text']])

#print(len(all_words))


from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()


not_disaster =' '.join([text for text in df['clean_text'][df['target'] == 0]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(not_disaster)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
disaster =' '.join([text for text in df['clean_text'][df['target'] == 1]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(disaster)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()
# function to collect hashtags

def hashtag_extract(x):

    hashtags = []

    # Loop over the words in the tweet

    for i in x:

        ht = re.findall(r"#(\w+)", i)

        hashtags.append(ht)



    return hashtags
# extracting hashtags from non Disaster tweets



HT_Not_Disaster = hashtag_extract(df['clean_text'][df['target'] == 0])



# extracting hashtags from Disaster tweets

HT_Disaster = hashtag_extract(df['clean_text'][df['target'] == 1])



# unnesting list

HT_Not_Disaster = sum(HT_Not_Disaster,[])

HT_Disaster = sum(HT_Disaster,[])
# Fake Disaster tweets



a = nltk.FreqDist(HT_Not_Disaster)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})

# selecting top 10 most frequent hashtags     

d = d.nlargest(columns="Count", n = 10) 

plt.figure(figsize=(16,5))

ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()
# True Disaster tweets



b = nltk.FreqDist(HT_Disaster)

e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})

# selecting top 10 most frequent hashtags

e = e.nlargest(columns="Count", n = 10)   

plt.figure(figsize=(16,5))

ax = sns.barplot(data=e, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

# bag-of-words feature matrix

bow = bow_vectorizer.fit_transform(df['clean_text'])
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

# TF-IDF feature matrix

tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])