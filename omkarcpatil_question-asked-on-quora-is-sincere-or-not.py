# all import statements

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

#from wordcloud import WordCloud as wc   # not needed

from nltk.corpus import stopwords

import matplotlib.pylab as pylab

import matplotlib.pyplot as plt

from pandas import get_dummies

import matplotlib as mpl

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import warnings

import sklearn

import string

import scipy

import numpy

import nltk

import json

import sys

import csv

import os
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
train = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')

test = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')

train.head()
print(train.info())

print(test.info())
# shape for train and test

print('Shape of train:',train.shape)

print('Shape of test:',test.shape)
# How many NA elements in every column!!

# Good news, it is Zero!

# To check out how many null info are on the dataset, we can use isnull().sum().

# recall from info() -> we found that it has zero Nulls. 



train.isnull().sum()



# data is infact clean and ready for use.
# in case , their were NA or None values in any row then we would drop the row.



# remove rows that have NA's

print('Before Droping',train.shape)

train = train.dropna()

print('After Droping',train.shape)
# Number of words in the text



train["num_words"] = train["question_text"].apply(lambda x: len(str(x).split()))

test["num_words"] = test["question_text"].apply(lambda x: len(str(x).split()))

print('maximum of num_words in train',train["num_words"].max())

print('min of num_words in train',train["num_words"].min())

print("maximum of  num_words in test",test["num_words"].max())

print('min of num_words in train',test["num_words"].min())
# Number of unique words in the text

train["num_unique_words"] = train["question_text"].apply(lambda x: len(set(str(x).split())))

test["num_unique_words"] = test["question_text"].apply(lambda x: len(set(str(x).split())))



print('maximum of num_unique_words in train',train["num_unique_words"].max())



print("maximum of num_unique_words in test",test["num_unique_words"].max())

# Number of stopwords in the text



#from nltk.corpus import stopwords

eng_stopwords = set(stopwords.words("english"))



train["num_stopwords"] = train["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

test["num_stopwords"] = test["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))



print('maximum of num_stopwords in train',train["num_stopwords"].max())

print("maximum of num_stopwords in test",test["num_stopwords"].max())

# Number of punctuations in the text



train["num_punctuations"] =train['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test["num_punctuations"] =test['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

print('maximum of num_punctuations in train',train["num_punctuations"].max())

print("maximum of num_punctuations in test",test["num_punctuations"].max())
# lets figure out how many unique target values exist.

# like we expect : 0 -> sincere qns and 1 -> un-sincere qns



# You see number of unique item for Target with command below:

train_target = train['target'].values



np.unique(train_target)

#train.where(train ['target']==1).count()

train[train.target==1].count()
# visualising the imbalance in data set



ax=sns.countplot(x='target',hue="target", data=train  ,linewidth=5,edgecolor=sns.color_palette("dark", 3))

plt.title('Is data set imbalance?');
# step 1: Change all the text to lower case. 



# This is required as python interprets 'quora' and 'QUORA' differently



train['question_text'] = [entry.lower() for entry in train['question_text']]



test['question_text'] = [entry.lower() for entry in test['question_text']]



train.head()
# more imports for NLP

from nltk.tokenize import word_tokenize

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import LabelEncoder

from collections import defaultdict

from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import model_selection, naive_bayes, svm

from sklearn.metrics import accuracy_score
# taking backup

trainbackup=train

testbackup=test

trainbackup.shape
# keeping only 2000 questions for analysis

train= train.head(2000)

test= test.head(2000)
# step 2 : Tokenization : In this each entry in the corpus will be broken 

#                         into set of words





train['question_text']= [word_tokenize(entry) for entry in train['question_text']]



test['question_text']= [word_tokenize(entry) for entry in test['question_text']]

train.head()
# Set random seed

# This is used to reproduce the same result every time 

# if the script is kept consistent otherwise each run 

# will produce different results. The seed can be set to any number.

np.random.seed(500)
## for train data



# step 3, 4 and 5

# Remove Stop words and Numeric data 

# and perfom Word Stemming/Lemmenting.



# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb

# or adjective etc. By default it is set to Noun

tag_map = defaultdict(lambda : wn.NOUN)

tag_map['J'] = wn.ADJ

tag_map['V'] = wn.VERB

tag_map['R'] = wn.ADV

# the tag_map would map any tag to 'N' (Noun) except

# Adjective to J, Verb -> v, Adverb -> R

# that means if you get a Pronoun then it would still be mapped to Noun





for index,entry in enumerate(train['question_text']):

    # Declaring Empty List to store the words that follow the rules for this step

    Final_words = []

    

    # Initializing WordNetLemmatizer()

    word_Lemmatized = WordNetLemmatizer()

    #print(help(pos_tag(entry)))

    # pos_tag function below will provide the 'tag' 

    # i.e if the word is Noun(N) or Verb(V) or something else.

    for word, tag in pos_tag(entry):

    

        # Below condition is to check for Stop words and consider only 

        # alphabets

        if word not in stopwords.words('english') and word.isalpha():

            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])

            

            Final_words.append(word_Final)

    #print(Final_words)        

    # The final processed set of words for each iteration will be stored 

    # in 'question_text_final'

    train.loc[index,'question_text_final'] = str(Final_words)

    

   
## for test data



# step 3, 4 and 5

# Remove Stop words and Numeric data 

# and perfom Word Stemming/Lemmenting.



# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb

# or adjective etc. By default it is set to Noun

tag_map = defaultdict(lambda : wn.NOUN)

tag_map['J'] = wn.ADJ

tag_map['V'] = wn.VERB

tag_map['R'] = wn.ADV

# the tag_map would map any tag to 'N' (Noun) except

# Adjective to J, Verb -> v, Adverb -> R

# that means if you get a Pronoun then it would still be mapped to Noun





for index,entry in enumerate(test['question_text']):

    # Declaring Empty List to store the words that follow the rules for this step

    Final_words_test = []

    

    # Initializing WordNetLemmatizer()

    word_Lemmatized = WordNetLemmatizer()

    

    # pos_tag function below will provide the 'tag' 

    # i.e if the word is Noun(N) or Verb(V) or something else.

    for word, tag in pos_tag(entry):

        # Below condition is to check for Stop words and consider only 

        # alphabets

        if word not in stopwords.words('english') and word.isalpha():

            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])

            Final_words_test.append(word_Final)

            

    # The final processed set of words for each iteration will be stored 

    # in 'question_text_final'

    test.loc[index,'question_text_final'] = str(Final_words_test)

    
test.head()
Tfidf_vect = TfidfVectorizer()

Tfidf_vect.fit(train['question_text_final'])



Train_X_Tfidf = Tfidf_vect.transform(train['question_text_final'])



Test_X_Tfidf = Tfidf_vect.transform(test['question_text_final'])
#print(Train_X_Tfidf)

print(Test_X_Tfidf[:4])
# You can use the below syntax to see the vocabulary that 

# it has learned from the corpus

print(Tfidf_vect.vocabulary_)




# fit the training dataset on the NB classifier

Naive = naive_bayes.MultinomialNB()



Naive.fit(Train_X_Tfidf,train['target'])



# predict the labels on validation dataset

predictions_NB = Naive.predict(Test_X_Tfidf)



# Use accuracy_score function to get the accuracy

#print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, train['target'])*100)

print(predictions_NB)



accuracy_score(predictions_NB,train.target)*100
# Classifier - Algorithm - SVM

# fit the training dataset on the classifier

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')



SVM.fit(Train_X_Tfidf,train['target'])



# predict the labels on validation dataset

predictions_SVM = SVM.predict(Test_X_Tfidf)



# Use accuracy_score function to get the accuracy

# print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, train['target'])*100)

print(predictions_SVM)

predictions_SVM[0]





accuracy_score(predictions_SVM,train.target)*100