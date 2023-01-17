### Import 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # data visualization

from sklearn.model_selection import train_test_split # tran and test data split

from sklearn.linear_model import LogisticRegression # Logistic Regression 

from sklearn.svm import SVC #Support Vector Machine 

from sklearn.ensemble import RandomForestClassifier # Random Rorest Classifier 

from sklearn.metrics import roc_auc_score # ROC and AUC 

from sklearn.metrics import accuracy_score # Accuracy 

from sklearn.metrics import recall_score # Recall 

from sklearn.metrics import precision_score # Prescison 

from sklearn.metrics import classification_report # Classification Score Report 

%matplotlib inline 

from sklearn.svm import SVC

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
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.target.value_counts()
location=train[train['target']==0]['location']
train.isna().sum()
train['keyword'].unique()
# top ten keywords

train['keyword'].value_counts()[:10].plot(kind='bar',figsize=(24,5),color=['lightblue','salmon']);
X=train.text.values

y=train.target.values
import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

import string

from sklearn.base import TransformerMixin

#Cleaning and preprocessing

# Create our list of punctuation marks

punctuations = string.punctuation



# Create our list of stopwords

nlp = spacy.load('en')

stop_words = spacy.lang.en.stop_words.STOP_WORDS



# Load English tokenizer, tagger, parser, NER and word vectors

parser = English()



# Creating our tokenizer function

def spacy_tokenizer(sentence):

    # Creating our token object, which is used to create documents with linguistic annotations.

    mytokens = parser(sentence)



    # Lemmatizing each token and converting each token into lowercase

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]



    # Removing stop words

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]



    # return a preprocessed list of tokens

    return mytokens



# Custom transformer using spaCy

class predictors(TransformerMixin):

    def transform(self, X, **transform_params):

        # Cleaning Text

        return [clean_text(text) for text in X]



    def fit(self, X, y=None, **fit_params):

        return self



    def get_params(self, deep=True):

        return {}



# Basic function to clean the text

def clean_text(text):

    # Removing spaces and converting text into lowercase

    return text.strip().lower()



# Splitting dataset in train and test

X_train, X_test, y_train, y_test = train_test_split(train.text, train.target, test_size=0.2)



#Train-test shape

#Train-test shape

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.pipeline import Pipeline
#Support Vector Machine Classifier

# Create pipeline using Bag of Words

pipe = Pipeline([("cleaner", predictors()),

                 ('vectorizer', CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,3))),

                 ('classifier', SVC())])



#Training the model.

pipe.fit(X_train,y_train)
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report, confusion_matrix

from wordcloud import WordCloud
# Predicting with a test dataset

y_pred = pipe.predict(X_test)



# Model Accuracy

print("Classification Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report\n")

print(classification_report(y_test, y_pred))

print("Confusion Matrix\n")

print(confusion_matrix(y_test, y_pred))
fig, ax = plt.subplots(figsize=(10, 10))

plot_confusion_matrix(pipe, X_test, y_test, values_format=' ', ax=ax) 

plt.title('Confusion Matrix')

plt.show()
from sklearn.naive_bayes import MultinomialNB

pipe = Pipeline([("cleaner", predictors()),

                 ('vectorizer', CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,3))),

                 ('clf', MultinomialNB())])



#Training the model.

pipe.fit(X_train,y_train)
# Predicting with a test dataset

y_pred = pipe.predict(X_test)



# Model Accuracy

print("Classification Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report\n")

print(classification_report(y_test, y_pred))

print("Confusion Matrix\n")

print(confusion_matrix(y_test, y_pred))
fig, ax = plt.subplots(figsize=(10, 10))

plot_confusion_matrix(pipe, X_test, y_test, values_format=' ', ax=ax) 

plt.title('Confusion Matrix')

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

X_train_counts.shape
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_tfidf.shape
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),

                      ('tfidf', TfidfTransformer()),

                      ('clf', MultinomialNB()),

])
import nltk

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):

        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),

                     ('tfidf', TfidfTransformer()),

                      ('mnb', MultinomialNB(fit_prior=False)),

 ])

text_mnb_stemmed = text_mnb_stemmed.fit(X_train, y_train)

predicted_mnb_stemmed = text_mnb_stemmed.predict(X_test)

np.mean(predicted_mnb_stemmed == y_test)

predicted_mnb_stemmed = text_mnb_stemmed.predict(test.text)

predicted_mnb_stemmed
submission['target'] = (predicted_mnb_stemmed > 0.5).astype(int)

submission
submission.to_csv("submission.csv", index=False, header=True)

fig, ax = plt.subplots(figsize=(10, 10))

plot_confusion_matrix(pipe, X_test, y_test, values_format=' ', ax=ax) 

plt.title('Confusion Matrix')

plt.show()