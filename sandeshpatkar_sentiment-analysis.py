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

import numpy as np

import pandas as pd

import time



# NLP

from nltk.corpus import stopwords

#for lemmatizing

from nltk.stem.wordnet import WordNetLemmatizer

import nltk

#importing n_grams

from nltk import ngrams

#importing count vectorizer

from sklearn.feature_extraction.text import CountVectorizer

#importing TFIDF Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



# ml

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')

df.head()
df.info()
df.shape
df.isnull().sum()
original_df_size = df.shape[0]
# removing duplicates



df = df.sort_values("ProductId",axis=0,ascending=True,inplace=False,kind="quicksort",na_position="last")

# Deduplication of the Entries

df = df.drop_duplicates(subset=["UserId","ProfileName","Time","Text"],keep="first",inplace=False)

new_df_size = df.shape[0]
print('Before removing duplicates:', original_df_size)

print('After removing duplicates:', new_df_size)

print('Number of duplicate values:', original_df_size - new_df_size)

print('Share of duplicate values:', ((original_df_size - new_df_size)/original_df_size)*100)
required_columns = ['Id', 'ProductId', 'HelpfulnessNumerator','HelpfulnessDenominator','Score','Summary','Text']

df = df[required_columns]
df['helpful_percent'] = df['HelpfulnessNumerator']/df['HelpfulnessDenominator']

df['helpful_percent'] = df['helpful_percent'].fillna(-1) #filling values where HelpfulnessDenominator = 0 leading to NaN
df[df['helpful_percent'] == -1].count()['helpful_percent']/len(df)
df['helpful_percent'].value_counts()
bin_labels = ['unavailable', '0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

bin_cuts = [-1, 0, 0.2, 0.4, 0.6, 0.8, 1.0]

df['helpful'] = pd.cut(df['helpful_percent'], bins = bin_cuts, 

                       labels = bin_labels, 

                       include_lowest = True)

df.head()
df['Score'].value_counts(normalize = True)*100
df1 = df[df['Score'] != 3]

df1.head()
to_0_1 = {1: 0, 2: 0, 4: 1, 5:1}

df1['target'] = df1['Score'].map(to_0_1)



df1.head()
df1['target'].value_counts()
X = df1['Text']

y = df1['target']

def predictor(x_data, y_data, m, tm, show_words = 'y'):

    # m: classification model

    # tm: text_model

    start_time=time.time()

    print('We are using:')

    print(m)

    print()

    print()

    X = tm.fit_transform(x_data)

    y = y_data

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

    m.fit(X_train, y_train)

    preds = m.predict(X_test)

    print("Score:", round(m.score(X_test, y_test)*100,2))

    print()

    print()

    print("Classification report:")

    print(classification_report(y_test, preds))

    print()

    print()

    if show_words == 'y':

        words = tm.get_feature_names()

        word_coeffs = list(m.coef_[0])

        word_coeff = pd.DataFrame({'words': words, 'coefficient': word_coeffs})

        word_coeff = word_coeff.sort_values(by = 'coefficient', ascending = False)

        print('Top 20 positive words:\n', word_coeff.head(20))

        print()

        print()

        print('Top 20 negative words:\n', word_coeff.tail(20))

    end_time=time.time()

    print('Time taken:', end_time - start_time)

    print()
lr = LogisticRegression() #default solver

lr1 = LogisticRegression(solver = 'newton-cg') #solver changed to newton-cg
#initialize CountVectorizer

cv = CountVectorizer(stop_words = 'english')
predictor(X, y, lr, cv)

predictor(X, y, lr1, cv)
import string

import re
df2 = df1.copy()
def clean_html(sentence):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', sentence)

    return cleantext



df2['Text'] = df2['Text'].apply(clean_html)
#to lower case

df2['Text'] = df2['Text'].str.lower()
puncs = list(string.punctuation)
# remove punctuation

for pun in puncs:

    df2['Text'] = df2['Text'].str.replace(pun, '')
X = df2['Text']

y = df2['target']
# fitting model after cleaning

cv = CountVectorizer(stop_words = 'english')

predictor(X, y, lr1, cv)
# lr = LogisticRegression(solver = 'newton-cg', penalty = 'l2', multi_class = 'multinomial', n_jobs = -1, max_iter = 10, class_weight = 'balanced')

#works
tfidf = TfidfVectorizer(stop_words = 'english')



predictor(X, y, lr1, tfidf)
ngram_tfidf = TfidfVectorizer(ngram_range = (1,2), stop_words = 'english')

predictor(X, y, lr1, ngram_tfidf, 'y')
ngram_tfidf = TfidfVectorizer(ngram_range = (1,3), stop_words = 'english')

predictor(X, y, lr1, ngram_tfidf, 'y')
tree = DecisionTreeClassifier(max_depth = 10, random_state = 1, max_features = None)
predictor(X, y, tree, tfidf, 'n')
lemma = WordNetLemmatizer()
def lemmatize_text(text):

    for w in nltk.tokenize.sent_tokenize(text):

        return lemma.lemmatize(w)
df3 = df2.copy()



start_time=time.time()

#lemmatizing text column

df3['Text'] = df3['Text'].apply(lemmatize_text)

end_time=time.time()

print('Time taken to lemmatize:', end_time - start_time)
df3['Text']
X = df3['Text']

y = df3['target']
predictor(X, y, lr1, tfidf)
