import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from nltk import word_tokenize



import os, re

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train.head()
### function to clean text (remove punctuation, links, lowercase all letters, etc)

def clean_text(text):

    temp = text.lower()

    temp = re.sub('\n', " ", temp)

    temp = re.sub('\'', "", temp)

    temp = re.sub('-', " ", temp)

    temp = re.sub(r"(http|https|pic.)\S+"," ",temp)

    temp = re.sub(r'[^\w\s]',' ',temp)

    

    return temp



### list of stop words that need to be removed

stop_words = ['as', 'in', 'of', 'is', 'are', 'were', 'was', 'it', 'for', 'to', 'from', 'into', 'onto', 

              'this', 'that', 'being', 'the','those', 'these', 'such', 'a', 'an']

### function to remove unnecessary words

def remove_stopwords(text):

    tokenized_words = word_tokenize(text)

    temp = [word for word in tokenized_words if word not in stop_words]

    temp = ' '.join(temp)

    return temp



### We save the cleaned and normalized texts in the new column, called 'clean'

train['clean'] = train['text'].apply(clean_text)

train['clean'] = train['clean'].apply(remove_stopwords)

train['clean']
def combine_attributes(text, keyword):

    var_list = [text, keyword]

    combined = ' '.join(x for x in var_list if x)

    return combined



train.fillna('', inplace=True)

train['combine'] = train.apply(lambda x: combine_attributes(x['clean'], x['keyword']), axis=1)
X = train['combine']

y = train['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer()



X_train_vect = vectorizer.fit_transform(X_train)

X_test_vect = vectorizer.transform(X_test)
from sklearn.svm import SVC

from sklearn.svm import LinearSVC



clf = SVC(kernel='linear')

clf.fit(X_train_vect, y_train)



y_pred = clf.predict(X_test_vect)
from sklearn.metrics import accuracy_score



accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix



confusion_matrix(y_test, y_pred)
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test.head()
### apply preprocessing on train data

train['clean'] = train['text'].apply(clean_text)

train['clean'] = train['clean'].apply(remove_stopwords)



### apply preprocessing on test data

test['clean'] = test['text'].apply(clean_text)

test['clean'] = test['clean'].apply(remove_stopwords)
train.fillna('', inplace=True)

train['combine'] = train.apply(lambda x: combine_attributes(x['clean'], x['keyword']), axis=1)



test.fillna('', inplace=True)

test['combine'] = test.apply(lambda x: combine_attributes(x['clean'], x['keyword']), axis=1)
X_train = train['combine']

y_train = train['target']



X_test = test['combine']
vectorizer = TfidfVectorizer()



X_train_vect = vectorizer.fit_transform(X_train)

X_test_vect = vectorizer.transform(X_test)
clf = SVC(kernel='linear')

clf.fit(X_train_vect, y_train)



y_pred = clf.predict(X_test_vect)
result = pd.DataFrame({'id':test['id'], 'target':y_pred})

result.to_csv('svm_submission.csv', index=False)
