# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
articles = pd.read_csv('/kaggle/input/bbc-fulltext-and-category/bbc-text.csv')

articles.head()
articles.describe()
articles['category'].unique()
articles.groupby('category').describe()
articles['length'] = articles['text'].apply(len)

articles.head(10)
articles['length'].plot(bins=50,kind='hist')
articles['length'].plot(bins=100,kind='hist')
articles['length'].max()
print(articles[articles['length'] == articles['length'].max()]['text'].iloc[0])
print(articles[articles['length'] == articles['length'].max()]['category'])
articles.hist(column='length', by='category', bins=50, figsize=(12,8))
articles.groupby('category').describe()
sns.boxplot(x='category',y='length',data=articles,palette='coolwarm')
sns.boxplot(x='category',y='length',data=articles,palette='coolwarm',showfliers=False)
import nltk
from nltk.corpus import stopwords
import string
def text_process(art):

    """

    Takes in a string of text, then perform the following:

    1. Remove all punctuations

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    """

    

    nopunc = [char for char in art if char not in string.punctuation]

    

    nopunc = ''.join(nopunc)

    

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(articles['text'])



print(len(bow_transformer.vocabulary_))
articles_bow = bow_transformer.transform(articles['text'])
print('Shape of Sparse Matrix: ', articles_bow.shape)

print('Amount of Non-Zero Occurences: ',articles_bow.nnz)
sparsity = (100.0 * articles_bow.nnz / (articles_bow.shape[0]*articles_bow.shape[1]))

print('sparsity: {}'.format(round(sparsity)))
from sklearn.feature_extraction.text import TfidfTransformer



tfidf_transformer = TfidfTransformer().fit(articles_bow)
articles_tfidf = tfidf_transformer.transform(articles_bow)

print(articles_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB



categorize_model = MultinomialNB().fit(articles_tfidf,articles['category'])
from sklearn.pipeline import Pipeline



pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=text_process)), # strings to token integer counts

    ('tfidf', TfidfTransformer()), # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()), # train on TF-IDF vectors w/ Naive Bayes classifier

])
from sklearn.model_selection import train_test_split



art_train, art_test, cat_train, cat_test = train_test_split(articles['text'],articles['category'])



print(len(art_train), len(art_test), len(cat_train) + len(cat_test))
pipeline.fit(art_train, cat_train)
predictions = pipeline.predict(art_test)
from sklearn.metrics import classification_report



print(classification_report(predictions,cat_test))