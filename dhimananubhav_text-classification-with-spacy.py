import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline
# Loading TSV file

df_amazon = pd.read_csv ("../input/amazon-alexa-reviews/amazon_alexa.tsv", sep="\t")
# Top 5 records

df_amazon.head()
# shape of dataframe

df_amazon.shape
# View data information

df_amazon.info()
# Feedback Value count

df_amazon.feedback.value_counts()
!pip install spacy
!python -m spacy download en
# Tokening the Data With spaCy

import spacy

import string

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English



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



    # return preprocessed list of tokens

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
bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
from sklearn.model_selection import train_test_split



X = df_amazon['verified_reviews'] # the features we want to analyze

ylabels = df_amazon['feedback'] # the labels, or answers, we want to test against



X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)
# Logistic Regression Classifier

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()



# Create pipeline using Bag of Words

pipe = Pipeline([("cleaner", predictors()),

                 ('vectorizer', bow_vector),

                 ('classifier', classifier)])



# model generation

pipe.fit(X_train,y_train)
from sklearn import metrics

# Predicting with a test dataset

predicted = pipe.predict(X_test)



# Model Accuracy

print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))

print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))

print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))