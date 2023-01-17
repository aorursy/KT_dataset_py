# scientific calculation and data analysis

import numpy as np

import pandas as pd

 

# visualization

import matplotlib.pyplot as plt

import seaborn as sns



# missingno

import missingno



# SKLearn

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

## Model

from sklearn.naive_bayes import GaussianNB



# NLP

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer



# WordCloud

from wordcloud import WordCloud



# basic imports

import string
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
test.head()
# train missing matrix

missingno.matrix(train)

plt.show()
# test missing matrix

missingno.matrix(test)

plt.show()
# missing bar plot train

missingno.bar(train)

plt.show()
# missing bar plot test

missingno.bar(test)

plt.show()
# null count in train and test

null_vals = pd.DataFrame(columns=['train', 'test'])

null_vals['train'] = train.isnull().sum()

null_vals['test'] = test.isnull().sum()

null_vals
# drop location features

train.drop('location', axis=1, inplace=True)
# drop nan rows

train.dropna(axis=0, inplace=True)
# check missing value in train dataset

train.isnull().sum()
fig, ax = plt.subplots(1,2)

sns.countplot(x='target', data=train, ax=ax[0])

ax[1].pie(train.target.value_counts(), labels=['Not Real', 'Real'], autopct='%1.1f%%')

plt.show()
# create remove_punctuation functionabs

def remove_punctuations(text):

    return "".join([c for c in text if c not in string.punctuation])
# Apply to text feature

train['text'] = train['text'].apply(lambda x: remove_punctuations(x))
# create lower_apha_num: convert to lower case and remove numerucal value

def lower_alpha_num(text):

    return [word for word in word_tokenize(text.lower()) if word.isalpha()]
# Apply lower_apha_num

train['text'] = train['text'].apply(lambda x: lower_alpha_num(x))
def remove_stopword(text):

    return [w for w in text if w not in stopwords.words('english')]
train['text'] = train['text'].apply(lambda x: remove_stopword(x))
train.head()
# Initiate Lamitizer

lemmatizer = WordNetLemmatizer()



def word_lemmatizer(text):

    lem_text = " ".join([lemmatizer.lemmatize(i) for i in text])

    return lem_text
train['text'] = train['text'].apply(lambda x: word_lemmatizer(x))
train.head()
X = train['text']

y = train['target']
vectorizer_tfidf = TfidfVectorizer()
X = vectorizer_tfidf.fit_transform(X)
pd.DataFrame(X.A, columns=vectorizer_tfidf.get_feature_names()).head()
X_train, X_val, y_train, y_val = train_test_split(X.A, y, test_size=0.3)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
classifier.score(X_train, y_train)
classifier.score(X_val, y_val)