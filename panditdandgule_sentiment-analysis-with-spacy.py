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

import spacy

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import sklearn as sk
data_yelp=pd.read_table('/kaggle/input/yelp_labelled.txt')

data_amazon=pd.read_table('/kaggle/input/amazon_cells_labelled.txt')

data_imdb=pd.read_table('/kaggle/input/imdb_labelled.txt')
#joining the tables

combined_col=[data_amazon,data_imdb,data_yelp]
#To observe how the data in each individual dataset is structured

print(data_amazon.columns)
#In order to add headers for columns in each dataset

for colname in combined_col:

    colname.columns=['Review','Label']
for colname in combined_col:

    print(colname.columns)
# In order to recognize which dataset belonged to which company, a 'Company' column is added as a key

company=['Amazon','imdb','yelp']

comb_data=pd.concat(combined_col,keys=company)
#Exploring the structure of the new data frame

print(comb_data.shape)
comb_data.head()
comb_data.sample(5)
comb_data.to_csv("Sentiment_Analysis_Dataset")
print(comb_data.columns)
print(comb_data.isnull().sum())
comb_data.isnull().sum().sum()
import spacy

import en_core_web_sm

from spacy.lang.en.stop_words import STOP_WORDS
nlp=en_core_web_sm.load()



#To build a list of stopwords

stopwords=list(STOP_WORDS)

print(stopwords)
import string

punctuations=string.punctuation



#Creating spacy parser

from spacy.lang.en import English

parser=English()
def my_tokenizer(sentence):

    mytokens=parser(sentence)

    mytokens=[word.lemma_.lower().strip() if word.lemma_!="-PRON-" else word.lower_ for word in mytokens]

    mytokens=[word for word in mytokens if word not in stopwords and word not in punctuations]

    return mytokens
#ML Packages

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.metrics import accuracy_score

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
#Custom transformer using Spacy

class predictors(TransformerMixin):

    def transform(self,X,**transform_params):

        return [clean_text(text) for text in X]

    def fit(self,X,y,**fit_params):

        return self

    def get_params(self,deep=True):

        return {}

    

#Basic function to clean the text

def clean_text(text):

    return text.strip().lower()
#Vectorization

vectorizer=CountVectorizer(tokenizer=my_tokenizer,ngram_range=(1,1))

classifier=LinearSVC()
#using Tfidf

tfvectorizer=TfidfVectorizer(tokenizer=my_tokenizer)
#splitting data set

from sklearn.model_selection import train_test_split
#Features and Labels

X=comb_data['Review']

ylabels=comb_data['Label']

X_train,X_test,y_train,y_test=train_test_split(X,ylabels,test_size=0.2,random_state=42)
pipe_countvect = Pipeline([("cleaner", predictors()),

                 ('vectorizer', vectorizer),

                 ('classifier', classifier)])

# Fit our data

pipe_countvect.fit(X_train,y_train)

# Predicting with a test dataset

sample_prediction = pipe_countvect.predict(X_test)



# Prediction Results

# 1 = Positive review

# 0 = Negative review

for (sample,pred) in zip(X_test,sample_prediction):

    print(sample,"Prediction=>",pred)

    

# Accuracy

print("Accuracy: ",pipe_countvect.score(X_test,y_test))

print("Accuracy: ",pipe_countvect.score(X_test,sample_prediction))

# Accuracy

print("Accuracy: ",pipe_countvect.score(X_train,y_train))