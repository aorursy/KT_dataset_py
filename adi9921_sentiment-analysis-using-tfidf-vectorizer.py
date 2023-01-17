# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os #interact withe file sysytem

import re #used for data cleaning

import nltk #used for data cleaing

from nltk.stem.porter import PorterStemmer #used for finding the root word of a word 

from nltk.corpus import stopwords #used fro removing the stop words

from bs4 import BeautifulSoup #used for removing the html tags in the text

from nltk.tokenize import TreebankWordTokenizer#it t

from sklearn.linear_model import LogisticRegression #for classification

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report #for testing the model accuracy
#importing all the negetive file first

train_neg=os.listdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/train/neg')#listing all the filenames present in the directory

os.chdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/train/neg')#setting the directory path

train_set=[]

for i in train_neg:

    fp=open(i,"r",encoding="utf8")

    train_set.append(fp.read())

    fp.close()
#now importing the all positive files

train_pos=os.listdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/train/pos')

os.chdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/train/pos')

for i in train_pos:

    fp=open(i,"r",encoding="utf8")

    train_set.append(fp.read())

    fp.close()

train_set[12500]
len(train_set)
train_set=pd.DataFrame(train_set,columns=["Reviews"])#converting train_set from list to a datframe

train_set["Rating"]=np.zeros([len(train_set),1],dtype=int)#adding rating column for all the rows to zero then we will change it to 1 for the positive reviews

train_set.loc[12500:25000,"Rating"]=1#changing the rating value to 1 for positive reviews

y=train_set.loc[:,"Rating"]
train_set[0:5]
train_set[12500:12505]
test_neg=os.listdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/test/neg')

test_pos=os.listdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/test/pos')
os.chdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/test/neg')

test_set=[]

for i in test_neg:

    fp=open(i,"r",encoding="utf8")

    test_set.append(fp.read())

    fp.close()

test_set[0]
os.chdir('/kaggle/input/imdb-movie-reviews-dataset/aclimdb/aclImdb/test/pos')

for i in test_pos:

    fp=open(i,"r",encoding="utf8")

    test_set.append(fp.read())

    fp.close()

test_set[12500]
test_set=pd.DataFrame(test_set,columns=["Reviews"])#converting test_set from list to a datframe

test_set["Rating"]=np.zeros([len(test_set),1],dtype=int)#adding rating column for all the rows to zero then we will change it to 1 for the positive reviews

test_set.loc[12500:25000,"Rating"]=1#changing the rating value to 1 for positive reviews

y_test=test_set.loc[ : ,"Rating"]
test_set[0:5]
test_set[12500:12505]
train_corpus=[]

tokenizer=TreebankWordTokenizer()

for i in range(0,len(train_set)):

    soup=BeautifulSoup(train_set["Reviews"][i],"html.parser")

    review=soup.get_text()#removes all html tags

    review.lower()#converting all words to lower case

    review=tokenizer.tokenize(review)#tokenizing using trebankwordtokenizer

    ps=PorterStemmer()

    review=[ps.stem(word) for word in review]#stemming the words

    review=' '.join(review)

    train_corpus.append(review)

train_corpus[0:10]
test_corpus=[]

for i in range(0,len(test_set)):

    soup=BeautifulSoup(test_set["Reviews"][i],"html.parser")

    review=soup.get_text()#removes all html tags

    review.lower()#converting all words to lower case

    review=tokenizer.tokenize(review)#tokenizing using trebankwordtokenizer

    ps=PorterStemmer()

    review=[ps.stem(word) for word in review]#stemming the words

    review=' '.join(review)

    test_corpus.append(review)
test_corpus[0:5]