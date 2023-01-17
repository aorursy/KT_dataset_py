# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import nltk

import string

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer



from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support as score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train_df.head()
test_df['target'] = sub['target']

test_df.head()
train_df.info()
test_df.info()
# For now drop the columns that may not be relevant



train_df.drop(['id', 'keyword', 'location'], axis=1, inplace=True)

test_df.drop(['id', 'keyword', 'location'], axis=1, inplace=True)
train_df.head()
test_df.head()
df = pd.concat([train_df, test_df], ignore_index = True)

df.head()
'''

Text data requires preparation before you can start using it for predictive modeling. The text preprocessing steps include

but are not limited to:



1. Removing punctuation

2. Tokenizing the text into words

3. Removing stopwords

4. Lemmatizing the word tokens

'''



stopwords = nltk.corpus.stopwords.words('english')

lemmatizer = WordNetLemmatizer()



def clean_text(text):

    text_clean = "".join([char for char in text if char not in string.punctuation])

    text_clean = re.split('\W+', text.lower())

    text_clean = [word for word in text_clean if word not in stopwords]

    text_clean = " ".join([lemmatizer.lemmatize(i, 'v') for i in text_clean])

    return text_clean
# Cleaning the 'text' column

df['text'] = df['text'].apply(lambda x: clean_text(x))

df.head()
# Initialize the Sentiment Analyzer function

sid = SentimentIntensityAnalyzer()



# Computing the sentiment score of the training data 'text' column

df['polarity'] = df['text'].apply(lambda x: sid.polarity_scores(x)['compound'])

'''

Converting the polarity score into classes:

class 0: Neutral

class 1: Positive

class 2: Negative

'''

df.loc[df['polarity'] > 0, 'polarity'] = 1

df.loc[df['polarity'] < 0, 'polarity'] = 2

df.head()
# TF-IDF vectorization on the training data



tfidf_vect = TfidfVectorizer()

X_tfidf = tfidf_vect.fit_transform(df['text'])



x_features = pd.concat([df['polarity'], pd.DataFrame(X_tfidf.toarray())], axis=1)

y_features = df['target']

x_features.head()
x_train, x_test, y_train, y_test = train_test_split(x_features, df['target'], test_size=0.3)
rf = RandomForestClassifier()

rf_model = rf.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)

precision, recall, fscore, support = score(y_test, y_pred, average = 'binary')

print("Precision: {} \nRecall: {} \nAccuracy: {}".format(round(precision, 3), round(recall, 3),

                                                        round((y_pred == y_test).sum()/len(y_pred), 3)))

def train_RF(n_est, depth):

        rf = RandomForestClassifier(n_estimators = n_est, max_depth = depth, n_jobs = -1)

        rf_model = rf.fit(x_train, y_train)

        y_pred = rf_model.predict(x_test)

        precision, recall, fscore, support = score(y_test, y_pred, average = 'binary')

        print('Est: {} / Depth: {} ------ Precision: {} / Recall: {} / Accuracy: {}'.format(n_est, depth, round(precision,3),

                                                                                           round(recall, 3), 

                                                                                        round((y_pred == y_test).sum()/len(y_pred), 3)))
for n_est in [10, 150, 300]:

    for depth in [30, 60, 90, None]:

        train_RF(n_est, depth)
sub['target'] = y_pred

sub.to_csv('submission.csv', index=False)

sub.head(3)