# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk #NLTK(Natural Language Toolkit)

from nltk.corpus import stopwords #NLTK(Natural Language Toolkit) in python has a list of stopwords stored in 16 different languages.

from nltk.tokenize import word_tokenize# To tokenize words

import re

import seaborn as sns

from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Any results you write to the current directory are saved as output.



train_df=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

## EDA

#Examine the dataset

train_df.head()

train_df.shape





train_df.head()
print (list(train_df))
#find out number of null values in location colum

print(train_df['location'].isnull().sum())

#Have a look at few colums value in the dataset

train_df['text'][0:2]

train_df['location'][0:2]
#Replace # with ''

spec_chars = ["#","%"]

for char in spec_chars:

    train_df['text'] = train_df['text'].str.replace(char, '')
train_df.head(20)
sns.heatmap(train_df.isnull(), cmap='viridis')
#Below step breaks up the strings into a list of words or pieces based on a specified pattern using Regular Expressions. The pattern I chose to use (r'\w') also removes punctuation and is a better option for this data in particular.
tokenizer = RegexpTokenizer(r'\w+')

train_df['text']=train_df['text'].apply(lambda x : tokenizer.tokenize(x.lower()))

##Removing stopwords with NTLK in python

stop_words=set(stopwords.words('english'))

def remove_stopwords(text):

    words= [w for w in text if w not in stopwords.words('english')]

    return words

#word_tokens = word_tokenize(train_df['text'])
train_df['text']=train_df['text'].apply(lambda x : remove_stopwords(x))

train_df['text'].head(10)
#Lemmatizing maps common words into one base. It returns a proper word that can be found in the dictionary.
#Dataset after removing stop words

train_df.head()
lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text):

    lem_text= [lemmatizer.lemmatize(i) for i in text]

    return lem_text 

train_df['text']=train_df['text'].apply(lambda x : word_lemmatizer(x))

train_df.head()
#Dataset after removing stop words

train_df['text'].head()
# Create y_train

y_train=train_df['target']


train_df['text']=train_df['text'].apply(lambda x: " ".join(x) )

count_vectorizer= feature_extraction.text.CountVectorizer() #AKA One-hot encoding

x=count_vectorizer.fit_transform(train_df['text'][0:5])



print(x[1].todense().shape) #There are 36 unique words (or "tokens") in the first five tweets.

print(x[1].todense())
#create vectors for all of our tweets.

train_vector=count_vectorizer.fit_transform(train_df['text'])

test_vector=count_vectorizer.transform(test_df['text'])





#Now we’re ready to fit a Multinomial Naive Bayes classifier model to our training data and use it to predict the test data’s labels:

naive_bayes = MultinomialNB()

naive_bayes.fit(train_vector, y_train)

#We are using f1.Generally, F1 Score is used when you want to seek a balance between Precision and Recall.

f1_scores = model_selection.cross_val_score(naive_bayes, train_vector, train_df["target"], cv=3, scoring="f1")

f1_scores
#Let's do predictions on our training set and build a submission for the competition.

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = naive_bayes.predict(test_vector)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)