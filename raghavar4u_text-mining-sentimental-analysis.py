%matplotlib inline  

# To make data visualisations display in Jupyter Notebooks 

import numpy as np   # linear algebra

import pandas as pd  # Data processing, Input & Output load



import matplotlib.pyplot as plt # Visuvalization & plotting

import seaborn as sns  #Data visualisation



import nltk # Natural Language Toolkit (statistical natural language processing (NLP) libraries )

from nltk.stem.porter import *   # Stemming 



from sklearn.model_selection import train_test_split, cross_val_score

                                    # train_test_split - Split arrays or matrices into random train and test subsets

                                    # cross_val_score - Evaluate a score by cross-validation



from sklearn.ensemble import RandomForestClassifier # RandomForestClassifier model to predict sentiment



from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer converts collection of text docs to a matrix of token counts



from sklearn.feature_extraction.text import TfidfTransformer # Converting occurrences to frequencies

from sklearn.feature_extraction.text import TfidfVectorizer



import warnings   # To avoid warning messages in the code run

warnings.filterwarnings("ignore")

train_MR = pd.read_csv("../input/train.tsv",sep="\t") # Train Moview Reviews 

test_MR = pd.read_csv("../input/test.tsv",sep="\t")

print(train_MR.shape)

print(test_MR.shape)

#train_MR.head()

test_MR.head()
train_MR.describe()
sns.countplot(data=train_MR,x='Sentiment')

dist = train_MR.groupby(["Sentiment"]).size()

print(dist)



dist_Percentage = round((dist / dist.sum())*100,2)

print(dist_Percentage)
train_MR['Length'] = train_MR['Phrase'].apply(lambda x: len(str(x).split(' ')))   ## WIll get the length of each phrase 

test_MR['Length'] = test_MR['Phrase'].apply(lambda x: len(str(x).split(' '))) 



train_MR.head()
train_MR.isnull().sum() 

test_MR.isnull().sum() 
train_MR['PreProcess_Sentence'] = train_MR['Phrase'].str.replace("[^a-zA-Z#]", " ")

test_MR['PreProcess_Sentence'] = test_MR['Phrase'].str.replace("[^a-zA-Z#]", " ")

train_MR.head()
train_MR['PreProcess_Sentence'] = train_MR['PreProcess_Sentence'].str.lower()

test_MR['PreProcess_Sentence'] = test_MR['PreProcess_Sentence'].str.lower()

test_MR['PreProcess_Sentence'].head()
count_vector = CountVectorizer()

train_counts = count_vector.fit_transform(train_MR['PreProcess_Sentence'])

train_counts.shape
count_vector.get_feature_names()
count_vector.vocabulary_.get('abdul')
## Term Frequencies (tf)

tf_transformer = TfidfTransformer(use_idf = False).fit(train_counts)  # Use fit() method to fit estimator to the data

train_tf = tf_transformer.transform(train_counts) # Use transform() method to transform count-matrix to 'tf' representation
## Term Frequency times Inverse Document Frequency (tf-idf)

tfidf_transformer = TfidfTransformer()

train_tfidf = tfidf_transformer.fit_transform(train_counts) # Use transform() method to transform count-matrix to 'tf-idf' representation
## Training a classifier to predict sentiment label of a phrase

# RandomForestClassifier model to predict sentiment



model = RandomForestClassifier()

Final_Model = model.fit(train_tfidf, train_MR['Sentiment'])
test_counts = count_vector.transform(test_MR['PreProcess_Sentence'])

# Use transform() method to transform test count-matrix to 'tf-idf' representation

test_tfidf = tfidf_transformer.transform(test_counts)

test_tfidf.shape
## Prediction on test data

predicted = Final_Model.predict(test_tfidf)
#for i, j in zip(testdata['PhraseId'], predicted):print(i, predicted[j])



for i, j in zip(test_MR['PhraseId'], predicted):

    print(i, predicted[j])

    

#testdata.head()

#testdata['PhraseId']

#predicted

# Writing *csv file for submission

import csv

with open('Movie_Sentiment.csv', 'w') as csvfile:

    csvfile.write('PhraseId,Sentiment\n')

    for i, j in zip(test_MR['PhraseId'], predicted):

         csvfile.write('{}, {}\n'.format(i, j))