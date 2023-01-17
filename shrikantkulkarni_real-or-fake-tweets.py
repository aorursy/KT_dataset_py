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
pwd
test_dataset = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
test_dataset.head()
train_dataset=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
train_dataset.head()
import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
sns.countplot(x="target",data=train_dataset)
from nltk.tokenize import sent_tokenize, word_tokenize

# Tokenizing

train_dataset['tokenized_text'] = train_dataset['text'].apply(word_tokenize)
train_dataset.head()
fake_tweets = train_dataset[train_dataset.target==0]

real_tweets = train_dataset[train_dataset.target==1]
fake = fake_tweets.tokenized_text.tolist()



fake_list = []

for sublist in fake:

    for item in sublist:

        fake_list.append(item)



real = real_tweets.tokenized_text.tolist()



real_list = []

for sublist in real:

    for item in sublist:

        real_list.append(item)

        

all_words = train_dataset.tokenized_text.tolist()



all_words_list = []

for sublist in all_words:

    for item in sublist:

        all_words_list.append(item)
import nltk
vocab_fake = nltk.FreqDist(fake_list)

vocab_real = nltk.FreqDist(real_list)

vocab_all = nltk.FreqDist(all_words_list)



print('Fake most common words: ',vocab_fake.most_common(20),'\n',

     'Real most common words: ',vocab_real.most_common(20),'\n',

     'All most common words: ',vocab_real.most_common(20))
fake_keywords = fake_tweets.keyword.tolist()
fake_list = []

for item in fake_keywords:

    fake_list.append(item)
fake_list
fake_keywords = fake_tweets.keyword.tolist()



fake_list = []

for item in fake_keywords:

    fake_list.append(item)



real_keywords = real_tweets.keyword.tolist()



real_list = []

for item in real_keywords:

    real_list.append(item)

        

all_keywords = train_dataset.keyword.tolist()



all_words_list = []

for item in all_keywords:

    all_words_list.append(item)
vocab_fake = nltk.FreqDist(fake_list)

vocab_real = nltk.FreqDist(real_list)

vocab_all = nltk.FreqDist(all_words_list)



print('Fake most common keywords: ',vocab_fake.most_common(20),'\n',

     'Real most common keywords: ',vocab_real.most_common(20),'\n',

     'All most common keywords: ',vocab_real.most_common(20))
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression


from nltk.corpus import stopwords

sw = stopwords.words('english')

new_words=('’','“', '”')

for i in new_words:

    sw.append(i)

sw
train_dataset['text']+ ' ' + train_dataset['keyword']
train_dataset = train_dataset.fillna("")

test_dataset = test_dataset.fillna("")

train_dataset['text'] = train_dataset['text']+ ' ' + train_dataset['keyword']

test_dataset['text'] = test_dataset['text']+ ' ' + test_dataset['keyword']
all_text=pd.concat([train_dataset[['text']], test_dataset[['text']]], ignore_index=True)
all_text.head()
test_dataset.head()
test_dataset.info()
## now testing with real test

vectorizer = TfidfVectorizer(stop_words=sw,lowercase=True)

x_all = vectorizer.fit_transform(all_text.text)
training_samples = train_dataset.shape[0]

X_train = x_all[:training_samples,:]

X_test = x_all[training_samples:,:]

y_train = train_dataset['target']
NB_classifier = MultinomialNB()

NB_classifier.fit(X_train,y_train)
pred = NB_classifier.predict(X_train)



print(classification_report(y_train,pred))
test_dataset['target'] = NB_classifier.predict(X_test)
test_dataset.head()
test_dataset.info()
test_dataset[['id', 'target']].to_csv(r'submission_sk.csv', index=False)