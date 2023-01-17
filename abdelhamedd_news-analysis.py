# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer , TfidfVectorizer

from nltk.stem import WordNetLemmatizer

from collections import Counter

from nltk.tokenize import word_tokenize

from nltk.corpus import wordnet

from nltk.corpus import stopwords

import re

import nltk

from nltk import pos_tag

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/news-articles/Articles.csv' , encoding= 'unicode_escape')
stop_words = stopwords.words('english')

normalizer = WordNetLemmatizer()
def get_part_of_speech(word):

  probable_part_of_speech = wordnet.synsets(word)

  pos_counts = Counter()

  pos_counts["n"] = len(  [ item for item in probable_part_of_speech if item.pos()=="n"]  )

  pos_counts["v"] = len(  [ item for item in probable_part_of_speech if item.pos()=="v"]  )

  pos_counts["a"] = len(  [ item for item in probable_part_of_speech if item.pos()=="a"]  )

  pos_counts["r"] = len(  [ item for item in probable_part_of_speech if item.pos()=="r"]  )

  most_likely_part_of_speech = pos_counts.most_common(1)[0][0]

  return most_likely_part_of_speech
def preprocess(article):

    cleaned = re.sub(r'\W+' , ' ' , article ).lower()

    tokenized = word_tokenize(cleaned)

    normalized = " ".join( [ normalizer.lemmatize( token , get_part_of_speech( token ) ) for token in tokenized if not re.match(r'\d+',token) and token not in stop_words])

    

    return normalized
df.head()
articles = []

for i in range(0 , df.shape[0]):

    articles.append(df['Article'].iloc[i])
preprcessed_articles = []

for i in articles :

    preprcessed_articles.append(preprocess(i))
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(preprcessed_articles)
try:

  article_index = [f"Article {i+1}" for i in range(len(articles))]

except:

  pass

try:

  feature_names = vectorizer.get_feature_names()

except:

  pass
try:

  df_word_counts = pd.DataFrame(counts.T.todense(), index = feature_names, columns = article_index)

  print(df_word_counts.head(10))

except:

  pass
transformer = TfidfTransformer(norm = None)
tfidf_scores_transformed = transformer.fit_transform(counts)
try:

  df_tf_idf = pd.DataFrame(tfidf_scores_transformed.T.todense(), index=feature_names, columns=article_index)

  print(df_tf_idf.head(10))

except:

  pass
vectorizer = TfidfVectorizer(norm = None)
tfidf_scores = vectorizer.fit_transform(preprcessed_articles)


try:

  df_tf_idf = pd.DataFrame( tfidf_scores.T.todense() , index=feature_names, columns=article_index)

  print(df_tf_idf.head( 10 ) )

except:

  pass

if np.allclose(tfidf_scores_transformed.todense(), tfidf_scores.todense()):

  print(pd.DataFrame({'Are the tf-idf scores the same?':['YES']}))

else:

  print(pd.DataFrame({'Are the tf-idf scores the same?':['No, something is wrong :(']}))
for i in range(1 , 20):

    print("Topic of  , " , df_tf_idf[[f'Article {i}']].idxmax())