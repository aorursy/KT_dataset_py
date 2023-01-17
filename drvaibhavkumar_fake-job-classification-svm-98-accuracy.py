#Importing Libraries

import re

import string

import numpy as np

import pandas as pd

import random

import missingno

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.base import TransformerMixin

from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report, confusion_matrix

from wordcloud import WordCloud

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

from sklearn.svm import SVC
#Reading dataset

data = pd.read_csv('../input/fake-job-postings/fake_job_postings.csv')
#Shape of the dataset

data.shape
#Head of the dataset

data.head()
data.interpolate(inplace=True)

data.isnull().sum()
#Fill NaN values with blank in the dataset

columns=['job_id', 'telecommuting', 'has_company_logo', 'has_questions', 'salary_range', 'employment_type']

for col in columns:

    del data[col]



data.fillna(' ', inplace=True)
data.head()
#Fraud and Real visualization

sns.countplot(data.fraudulent).set_title('Real & Fradulent')

data.groupby('fraudulent').count()['title'].reset_index().sort_values(by='title',ascending=False)
#Visualize job postings by countries

def split(location):

    l = location.split(',')

    return l[0]



data['country'] = data.location.apply(split)



country = dict(data.country.value_counts()[:11])

del country[' ']

plt.figure(figsize=(8,6))

plt.title('Country-wise Job Posting', size=20)

plt.bar(country.keys(), country.values())

plt.ylabel('No. of jobs', size=10)

plt.xlabel('Countries', size=10)
#Visualize the required experiences in the jobs

experience = dict(data.required_experience.value_counts())

del experience[' ']

plt.figure(figsize=(10,5))

plt.bar(experience.keys(), experience.values())

plt.title('No. of Jobs with Experience')

plt.xlabel('Experience', size=10)

plt.ylabel('No. of jobs', size=10)

plt.xticks(rotation=35)

plt.show()
#Most frequent jobs

print(data.title.value_counts()[:10])
#Titles and count of fraudulent jobs

print(data[data.fraudulent==1].title.value_counts()[:10])
#Titles and count of real jobs

print(data[data.fraudulent==0].title.value_counts()[:10])
#combine text in a single column to start cleaning our data

data['text']=data['title']+' '+data['location']+' '+data['company_profile']+' '+data['description']+' '+data['requirements']+' '+data['benefits']

del data['title']

del data['location']

del data['department']

del data['company_profile']

del data['description']

del data['requirements']

del data['benefits']

del data['required_experience']

del data['required_education']

del data['industry']

del data['function']

del data['country']
data.head()
#Separate fraud and actual jobs

fraudjobs_text = data[data.fraudulent==1].text

actualjobs_text = data[data.fraudulent==0].text
#Fradulent jobs wordcloud

STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS

plt.figure(figsize = (16,14))

wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(str(" ".join(fraudjobs_text)))

plt.imshow(wc,interpolation = 'bilinear')
#Actual jobs wordcloud

plt.figure(figsize = (16,14))

wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(str(" ".join(actualjobs_text)))

plt.imshow(wc,interpolation = 'bilinear')
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
# Splitting dataset in train and test

X_train, X_test, y_train, y_test = train_test_split(data.text, data.fraudulent, test_size=0.3)
#Train-test shape

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
#Support Vector Machine Classifier



# Create pipeline using Bag of Words

pipe = Pipeline([("cleaner", predictors()),

                 ('vectorizer', CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,3))),

                 ('classifier', SVC())])



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