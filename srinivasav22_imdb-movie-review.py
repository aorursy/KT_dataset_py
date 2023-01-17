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
import numpy as np

import pandas as pd



train = pd.read_excel('../input/imdb-movie-review-nlp-project/Data_Train.xlsx')
train.head()
train.shape
#Printing the dataset info

print(train.info)
#Printing the group by description of each category

train.groupby("SECTION").describe()
#Removing duplicates to avoid overfitting



train.drop_duplicates(inplace=True)
import nltk

from nltk.corpus import stopwords

import string




#A punctuations string for reference (added other valid characters from the dataset)

all_punctuations = string.punctuation + '‘’,:”][],' 



#Method to remove punctuation marks from the data

def punc_remover(raw_text):

    no_punct = "".join([i for i in raw_text if i not in all_punctuations])

    return no_punct



#Method to remove stopwords from the data

def stopword_remover(no_punc_text):

    words = no_punc_text.split()

    no_stp_words = " ".join([i for i in words if i not in stopwords.words('english')])

    return no_stp_words



#Method to lemmatize the words in the data

lemmer = nltk.stem.WordNetLemmatizer()

def lem(words):

    return " ".join([lemmer.lemmatize(word,'v') for word in words.split()])



#Method to perform a complete cleaning

def text_cleaner(raw):

    cleaned_text = stopword_remover(punc_remover(raw))

    return lem(cleaned_text)
#Applying the cleaner method to the entire data

train['CLEAN_STORY'] = train['STORY'].apply(text_cleaner)



#Checking the new dataset

print(train.values) 
#Creating TF-IDF Vectors

#Importing TfidfTransformer from sklearn

from sklearn.feature_extraction.text import TfidfTransformer



#Fitting the bag of words data to the TF-IDF transformer

tfidf_transformer = TfidfTransformer().fit(bow)



#Transforming the bag of words model to TF-IDF vectors

storytfidf = tfidf_transformer.transform(bow)
#Importing sklearn’s Countvectorizer

from sklearn.feature_extraction.text import CountVectorizer



#Creating a bag-of-words dictionary of words from the data

bow_dictionary = CountVectorizer().fit(train['CLEAN_STORY'])



#Total number of words in the bow_dictionary

len(bow_dictionary.vocabulary_)



#Using the bow_dictionary to create count vectors for the cleaned data.

bow = bow_dictionary.transform(train['CLEAN_STORY'])



#Printing the shape of the bag of words model

print(bow.shape)
#Creating a Multinomial Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB



#Fitting the training data to the classifier

classifier = MultinomialNB().fit(storytfidf, train['SECTION'])
#Importing and cleaning the test data

test = pd.read_excel('../input/imdb-movie-review-nlp-project/Data_Test.xlsx')

test['CLEAN_STORY'] = test['STORY'].apply(text_cleaner)



#Printing the cleaned data

print(test.values)
#Importing the Pipeline module from sklearn

from sklearn.pipeline import Pipeline



#Initializing the pipeline with necessary transformations and the required classifier

pipe = Pipeline([

    ('bow', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('classifier', MultinomialNB())])
#Fitting the training data to the pipeline

pipe.fit(train['CLEAN_STORY'], train['SECTION'])



#Predicting the SECTION 

y_pred = pipe.predict(test['CLEAN_STORY'])



#Writing the predictions to an excel sheet

pd.DataFrame(y_pred, columns = ['SECTION']).to_excel("predictions.xlsx")