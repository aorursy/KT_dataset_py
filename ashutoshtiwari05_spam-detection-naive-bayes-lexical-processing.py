# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_email = pd.read_csv("../input/spam.csv", encoding="latin-1")
df_email.shape
df_email.head()
email_dataset = []

for index,row in df_email.iterrows():

    email_dataset.append((row['v2'],row['v1']))
print(email_dataset[:5])
print(len(email_dataset))
spam_count = len(df_email[df_email['v1'] == 'spam'])

print((spam_count/len(email_dataset)) * 100)
ham_count = len(df_email[df_email['v1'] == 'ham'])

print((ham_count/len(email_dataset)) * 100)
stemmer = PorterStemmer()

wordnet_lemmatizer = WordNetLemmatizer()
def preprocess(document, stem=True):

    #following changes document to lower case, 

    #and removes stopwords and lemmatizes/stems the remainder of the sentence'



    # change sentence to lower case

    document = document.lower()



    # tokenize the given document into words

    words = word_tokenize(document)



    # remove stop words

    words = [word for word in words if word not in stopwords.words("english")]



    if stem:

        words = [stemmer.stem(word) for word in words]

    else:

        words = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]



    # join words to make sentence

    document = " ".join(words)



    return document
# Performing preprocessing on the messages in the data

messages_set =[]

for(message,label) in email_dataset :

    filtered_words = [e.lower() for e in preprocess(message, stem= False).split() if len(e) >3]

    messages_set.append((filtered_words, label))
print(messages_set[:5])
## creating a consolidation list of all the messages we have in the message set

## in order to select the word feature list in upcoming steps.



def get_messages(messages) :

    words =[]

    for(message,lable) in messages :

        words.extend(message)

    return words
## now creating the word feature list using the FreqDist function

## FreqDist Method : returns the frequency of the word by calculating the tf-idf scores



def word_featurs(wordlist) :

    wordlist = nltk.FreqDist(wordlist)

    word_features = wordlist.keys()

    return word_features
word_features = word_featurs(get_messages(messages_set))

print(len(word_features))
index = int((len(messages_set) * 0.7))
import random

random.shuffle(messages_set)
train_messages,test_messages= messages_set[:index],messages_set[index:]
print(len(messages_set))

print((len(train_messages)/len(messages_set)) * 100)  # 70%

print((len(test_messages)/len(messages_set)) * 100)   # 30%
def extract_features(document) :

    word_documents = set(document)

    features = {}

    for word in word_features:

        features['contains(%s)' % word] = (word in word_documents)

    return features
training_set = nltk.classify.apply_features(extract_features,train_messages)

test_set = nltk.classify.apply_features(extract_features,test_messages)
print(training_set[:5])
spam_Classifier = nltk.NaiveBayesClassifier.train(training_set)
print(nltk.classify.accuracy(spam_Classifier, training_set))
print(nltk.classify.accuracy(spam_Classifier, test_set))
### Printing the important features in the classifier evaluated above

print(spam_Classifier.show_most_informative_features(50))



### The below probability "179 against 1 says that award in a 

### given message has very high probability to be a SPAM"