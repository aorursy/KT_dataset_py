# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
import time
import datetime
from wordcloud import WordCloud
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV
%matplotlib inline
trainv=pd.read_csv('../input/nlp-getting-started/train.csv')
testv=pd.read_csv('../input/nlp-getting-started/test.csv')
trainv['Word Count']=trainv['text'].apply(len)
trainv.head()
trainv.shape
testv.shape
#Missing values in train set
trainv.isnull().sum()
#Counting the number of events and classifying them as 1:real disaster and 0: not a disaster.
trainv['target'].value_counts()
#Displaying the Target Distribution
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), dpi=100)
sns.countplot(trainv['target'], ax=axes[0])
axes[1].pie(trainv['target'].value_counts(),
            labels=['Not Disaster', 'Disaster'],
            autopct='%1.2f%%',
            shadow=True,
            explode=(0.06, 0),
            startangle=60)
fig.suptitle('Distribution of the Tweets', fontsize=24)
plt.show()
g=sns.FacetGrid(trainv,col='target')
g.map(plt.hist,'Word Count',bins=50)
sns.barplot(y=trainv['keyword'].value_counts()[:30].index,x=trainv['keyword'].value_counts()[:30])
disaster_tweets = trainv[trainv['target']==1]['text']
disaster_tweets.values[5]
non_disaster_tweets = trainv[trainv['target']==0]['text']
non_disaster_tweets.values[1]
fig, (ax1) = plt.subplots(1,figsize=[20, 8])
wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(disaster_tweets))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Disaster Tweets',fontsize=30);
fig, (ax2) = plt.subplots(1,figsize=[20, 8])
wordcloud2 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(non_disaster_tweets))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Non Disaster Tweets',fontsize=30);
def cleaner(text):
   #Make text lowercase, remove text in square brackets,remove links,remove punctuation and remove words containing numbers.
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = text.lower()
    return text

# Applying the cleaning function to both test and training datasets
trainv['text'] = trainv['text'].apply(lambda x: cleaner(x))
testv['text'] = testv['text'].apply(lambda x: cleaner(x))

# Let's take a look at the updated text
trainv['text'].head()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
trainv['text'] = trainv['text'].apply(lambda x: tokenizer.tokenize(x))
testv['text'] = testv['text'].apply(lambda x: tokenizer.tokenize(x))
trainv['text'].head()
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words


trainv['text'] = trainv['text'].apply(lambda x : remove_stopwords(x))
testv['text'] = testv['text'].apply(lambda x : remove_stopwords(x))
trainv.head()
# The text format after preprocessing 
def combiner(list_of_text):
    text = ' '.join(list_of_text)
    return text

trainv['text'] = trainv['text'].apply(lambda x : combiner(x))
testv['text'] = testv['text'].apply(lambda x : combiner(x))
trainv['text']
trainv.head()
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(trainv['text'])
test_vectors = count_vectorizer.transform(testv["text"])

## Keeping only non-zero elements to preserve space 
print(train_vectors[3].todense())
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(trainv['text'])
test_tfidf = tfidf.transform(testv["text"])
# Fitting a simple Naive Bayes on Counts
clf_NB = MultinomialNB()
scores = model_selection.cross_val_score(clf_NB, train_vectors, trainv["target"], cv=5, scoring="f1")
scores
clf_NB.fit(train_vectors,trainv["target"])
# Fitting a simple Naive Bayes on TFIDF
clf_NB_TFIDF = MultinomialNB()
scores = model_selection.cross_val_score(clf_NB_TFIDF, train_tfidf, trainv["target"], cv=5, scoring="f1")
scores
clf_NB_TFIDF.fit(train_tfidf, trainv["target"])
def submission(submission_file_path,model,test_vectors):
    sample_submission = pd.read_csv(submission_file_path)
    sample_submission["target"] = model.predict(test_vectors)
    sample_submission.to_csv("submission.csv", index=False)
submission_file_path = "../input/nlp-getting-started/sample_submission.csv"
test_vectors=test_tfidf
submission(submission_file_path,clf_NB_TFIDF,test_vectors)