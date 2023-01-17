# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from nltk.stem import PorterStemmer

from nltk.stem import LancasterStemmer

#create an object of class PorterStemmer

porter = PorterStemmer()

lancaster=LancasterStemmer()

#provide a word to be stemmed

print("Porter Stemmer")

print(porter.stem("cats"))

print(porter.stem("trouble"))

print(porter.stem("troubling"))

print(porter.stem("troubled"))

print("Lancaster Stemmer")

print(lancaster.stem("cats"))

print(lancaster.stem("trouble"))

print(lancaster.stem("troubling"))

print(lancaster.stem("troubled"))
from nltk.tokenize import sent_tokenize, word_tokenize

def stemSentence(sentence):

    token_words=word_tokenize(sentence)

    token_words

    stem_sentence=[]

    for word in token_words:

        stem_sentence.append(porter.stem(word))

        stem_sentence.append(" ")

    return "".join(stem_sentence)



sentence="Pythoners are very intelligent and work very pythonly and now they are pythoning their way to success."

x=stemSentence(sentence)

print(x)

import nltk

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()



sentence = "He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun."

punctuations="?:!.,;"

sentence_words = nltk.word_tokenize(sentence)

for word in sentence_words:

    if word in punctuations:

        sentence_words.remove(word)



sentence_words

print("{0:20}{1:20}".format("Word","Lemma"))

for word in sentence_words:

    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word)))

for word in sentence_words:

    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word, pos="v")))

from nltk.tokenize import word_tokenize

# Import Counter

from collections import Counter



# Read text files

f = open("../input/textdb/articles.txt", "r")

article = f.read()

#print()

# Tokenize the article: tokens

tokens = word_tokenize(article)



# Convert the tokens into lowercase: lower_tokens

lower_tokens = [t.lower() for t in tokens]



# Create a Counter with the lowercase tokens: bow_simple

bow_simple = Counter(lower_tokens)



# Print the 10 most common tokens

print(bow_simple.most_common(10))

# Import WordNetLemmatizer

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

# Import Counter

from collections import Counter



# Read text files

f = open("../input/textdb/articles.txt", "r")

article = f.read()

#print()

# Tokenize the article: tokens

tokens = word_tokenize(article)



# English Stop words

english_stops = set(stopwords.words('english'))



# Convert the tokens into lowercase: lower_tokens

lower_tokens = [t.lower() for t in tokens]



# Retain alphabetic words: alpha_only

alpha_only = [t for t in lower_tokens if t.isalpha()]



# Remove all stop words: no_stops

no_stops = [t for t in alpha_only if t not in english_stops]



# Instantiate the WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()



# Lemmatize all tokens into a new list: lemmatized

lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]



# Create the bag-of-words: bow

bow = Counter(lemmatized)



# Print the 10 most common tokens

print(bow.most_common(10))

import nltk

# Import WordNetLemmatizer

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

# Import Counter

from collections import Counter



# Read text files

f = open("../input/textdb/articles.txt", "r")

article = f.read()

#print()



# Tokenize the article into sentences: sentences

sentences = nltk.sent_tokenize(article)



# Tokenize each sentence into words: token_sentences

token_sentences = [nltk.word_tokenize(sent) for sent in sentences]



# Tag each tokenized sentence into parts of speech: pos_sentences

pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 



# Create the named entity chunks: chunked_sentences

chunked_sentences = nltk.ne_chunk_sents(pos_sentences,binary=True)



# Test for stems of the tree with 'NE' tags

for sent in chunked_sentences:

    for chunk in sent:

        if hasattr(chunk, "label") and chunk.label() == "NE":

            print(chunk)

import nltk

# Import WordNetLemmatizer

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

# Import Counter

from collections import Counter

from collections import defaultdict

import matplotlib.pyplot as plt



# Read text files

f = open("../input/textdb/articles.txt", "r")

article = f.read()



# Tokenize the article into sentences: sentences

sentences = nltk.sent_tokenize(article)



# Tokenize each sentence into words: token_sentences

token_sentences = [nltk.word_tokenize(sent) for sent in sentences]



# Tag each tokenized sentence into parts of speech: pos_sentences

pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 



# Create the named entity chunks: chunked_sentences

chunked_sentences = nltk.ne_chunk_sents(pos_sentences,binary=True)



# Create the defaultdict: ner_categories

ner_categories = defaultdict(int)



# Create the nested for loop

for sent in chunked_sentences:

    for chunk in sent:

        if hasattr(chunk, 'label'):

            ner_categories[chunk.label()] += 1

            

# Create a list from the dictionary keys for the chart labels: labels

labels = list(ner_categories.keys())



# Create a list of the values: values

values = [ner_categories.get(l) for l in labels]



# Create the pie chart

plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)



# Display the chart

plt.show()

# Import spacy

import spacy



# Read text files

f = open("../input/textdb/articles.txt", "r")

article = f.read()



# Instantiate the English model: nlp

nlp = spacy.load('en',tagger=False, parser=False, matcher=False)



# Create a new document: doc

doc = nlp(article)



# Print all of the found entities and their labels

for ent in doc.ents:

    print(ent.text, ent.label_)

# Import the necessary modules

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

import pandas as pd

#importing csv

df = pd.read_csv('../input/textdb3/fake_or_real_news.csv')

# Print the head of df

print(df.head())



# Create a series to store the labels: y

y = df.label



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(df['text'],y,test_size=0.33, random_state=53)



# Initialize a CountVectorizer object: count_vectorizer

count_vectorizer = CountVectorizer(stop_words='english')



# Transform the training data using only the 'text' column values: count_train 

count_train = count_vectorizer.fit_transform(X_train.values)



# Transform the test data using only the 'text' column values: count_test 

count_test = count_vectorizer.transform(X_test.values)



# Print the first 10 features of the count_vectorizer

print(count_vectorizer.get_feature_names()[:10])

# Import the necessary modules

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

import pandas as pd

#importing csv

df = pd.read_csv('../input/textdb3/fake_or_real_news.csv')

# Print the head of df

print(df.head())



# Create a series to store the labels: y

y = df.label



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(df['text'],y,test_size=0.33, random_state=53)





# Import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



# Initialize a TfidfVectorizer object: tfidf_vectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)



# Transform the training data: tfidf_train 

tfidf_train = tfidf_vectorizer.fit_transform(X_train)



# Transform the test data: tfidf_test 

tfidf_test = tfidf_vectorizer.transform(X_test)



# Print the first 10 features

print(tfidf_vectorizer.get_feature_names()[:10])



# Print the first 5 vectors of the tfidf training data

print(tfidf_train.A[:5])

# Import the necessary modules

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

import pandas as pd

# Import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



#importing csv

df = pd.read_csv('../input/textdb3/fake_or_real_news.csv')

# Print the head of df

print(df.head())



# Create a series to store the labels: y

y = df.label



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(df['text'],y,test_size=0.33, random_state=53)





# Initialize a TfidfVectorizer object: tfidf_vectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)



# Transform the training data: tfidf_train 

tfidf_train = tfidf_vectorizer.fit_transform(X_train)



# Transform the test data: tfidf_test 

tfidf_test = tfidf_vectorizer.transform(X_test)



# Print the first 10 features

print(tfidf_vectorizer.get_feature_names()[:10])



# Print the first 5 vectors of the tfidf training data

print(tfidf_train.A[:5])



# Import the necessary modules

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

import pandas as pd

# Import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics



#importing csv

df = pd.read_csv('../input/textdb3/fake_or_real_news.csv')

# Print the head of df

print(df.head())



# Create a series to store the labels: y

y = df.label



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(df['text'],y,test_size=0.33, random_state=53)





# Initialize a TfidfVectorizer object: tfidf_vectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)



# Transform the training data: tfidf_train 

tfidf_train = tfidf_vectorizer.fit_transform(X_train)



# Transform the test data: tfidf_test 

tfidf_test = tfidf_vectorizer.transform(X_test)



# Create a Multinomial Naive Bayes classifier: nb_classifier

nb_classifier = MultinomialNB()



# Fit the classifier to the training data

nb_classifier.fit(tfidf_train,y_train)



# Create the predicted tags: pred

pred = nb_classifier.predict(tfidf_test)



# Calculate the accuracy score: score

score = metrics.accuracy_score(y_test,pred)

print("Accuracy Score : ",score)



# Calculate the confusion matrix: cm

cm = metrics.confusion_matrix(y_test,pred, labels=['FAKE', 'REAL'])

print("Confusion Matrix : \n",cm)



# Inspecting your model



print('Inspecting your model')



# Get the class labels: class_labels

class_labels = nb_classifier.classes_



# Extract the features: feature_names

feature_names = tfidf_vectorizer.get_feature_names()



# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights

feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))



# Print the first class label and the top 20 feat_with_weights entries

print('First class label and the top 20 feat_with_weights entries')

print(class_labels[0], feat_with_weights[:20])



# Print the second class label and the bottom 20 feat_with_weights entries

print('Second class label and the bottom 20 feat_with_weights entries')

print(class_labels[1], feat_with_weights[-20:])


