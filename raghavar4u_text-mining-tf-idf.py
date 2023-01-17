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



from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



import warnings   # To avoid warning messages in the code run

warnings.filterwarnings("ignore")
train_MR = pd.read_csv("../input/train.tsv",sep="\t") # Train Moview Reviews 

test_MR = pd.read_csv("../input/test.tsv",sep="\t")

print(train_MR.shape)

print(test_MR.shape)

train_MR.head()
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
train_MR['cat'] = 'TRAIN'

test_MR['cat'] = 'TEST'
train_test = train_MR.append(test_MR, ignore_index=True)

train_test.head()
train_test['PreProcess_Sentence'] = train_test['Phrase'].str.replace("[^a-zA-Z#]", " ")
train_test['PreProcess_Sentence'] = train_test['PreProcess_Sentence'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

train_test.head()
train_test['PreProcess_Sentence'] = train_test['PreProcess_Sentence'].str.lower()

train_test['PreProcess_Sentence'].head()
train_test['tokenized_words'] = train_test['PreProcess_Sentence'].apply(lambda x: x.split())

train_test.tokenized_words.head()
stemming = PorterStemmer() 

train_test['tokenized_words'] = train_test.tokenized_words.apply(lambda x: [stemming.stem(i) for i in x]) # stemming

print(train_test.tokenized_words.head())
corpus = train_test.PreProcess_Sentence  ## Collection of documents 

vectorizer = TfidfVectorizer(stop_words='english',analyzer='word')

print(vectorizer)



X = vectorizer.fit_transform(corpus)

print(X) 
idf = vectorizer.idf_

print(idf)
vectorizer.vocabulary_
vectorizer.get_feature_names()
col = ['feat_'+ i for i in vectorizer.get_feature_names()]

print(col[1:5])

print(X[1:5])

##tfidf = pd.DataFrame(X.todense(), columns=col) -- Memory issues

##tfidf

rr = dict(zip(vectorizer.get_feature_names(), idf))

token_weight = pd.DataFrame.from_dict(rr, orient='index').reset_index()

token_weight.columns=('token','weight')

token_weight = token_weight.sort_values(by='weight', ascending=False)

token_weight 