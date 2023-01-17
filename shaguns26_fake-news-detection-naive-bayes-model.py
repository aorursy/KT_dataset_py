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
truenews = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fakenews = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
fakenews.head()
truenews.head()
fakenews.describe()
truenews.describe()
truenews['True/Fake']='True'
fakenews['True/Fake']='Fake'
# Combine the 2 DataFrames into a single data frame
news = pd.concat([truenews, fakenews])
news["Article"] = news["title"] + news["text"]
news.sample(frac = 1) #Shuffle 100%
#Data Cleaning
from nltk.corpus import stopwords
import string
def process_text(s):

    # Check string to see if they are a punctuation
    nopunc = [char for char in s if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Convert string to lowercase and remove stopwords
    clean_string = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_string
# Tokenize the Article
#rerun, takes LOOOONG
news['Clean Text'] = news['Article'].apply(process_text)
news.sample(5)

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=process_text).fit(news['Clean Text'])

print(len(bow_transformer.vocabulary_)) #Total vocab words
#Bag-of-Words (bow) transform the entire DataFrame of text
news_bow = bow_transformer.transform(news['Clean Text'])
print('Shape of Sparse Matrix: ', news_bow.shape)
print('Amount of Non-Zero occurences: ', news_bow.nnz)
sparsity = (100.0 * news_bow.nnz / (news_bow.shape[0] * news_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))
#TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(news_bow)
news_tfidf = tfidf_transformer.transform(news_bow)
print(news_tfidf.shape)
#Train Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
fakenews_detect_model = MultinomialNB().fit(news_tfidf, news['True/Fake'])

#Model Evaluation
predictions = fakenews_detect_model.predict(news_tfidf)
print(predictions)
from sklearn.metrics import classification_report
print (classification_report(news['True/Fake'], predictions))
from sklearn.model_selection import train_test_split

news_train, news_test, text_train, text_test = train_test_split(news['Article'], news['True/Fake'], test_size=0.3)

print(len(news_train), len(news_test), len(news_train) + len(news_test))
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=process_text)),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', MultinomialNB()),  
])
pipeline.fit(news_train,text_train)
prediction = pipeline.predict(news_test)
print(classification_report(prediction,text_test))