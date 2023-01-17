#importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import nltk

import nltk.corpus

import string

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords

import re

import string

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier
print(os.listdir("../input"))
#reading the corpora and displaying it

f = open('/kaggle/input/sms-spam-collection-dataset/spam.csv', mode='r', encoding='latin-1')

message = []

for line in f.readlines():

    message.append(line.rstrip('\n'))
message[0:5]
print('The number of messages are {}'.format(len(message)))
#Importing the dataset

data= pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')



data = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

data= data.rename(columns = {'v1':'label','v2':'message'})

data.head()
#EDA

data.describe()
#Plot for label Data

print(data['label'].value_counts())

sns.set_style(style='darkgrid')

sns.countplot(data['label'], hue=data['label'])

plt.legend()

plt.show()
#gropuby for label

data.groupby(by='label').describe()
#Checking for missing values

data.isna().sum()
#Feature Engineering

data['length'] = data['message'].apply(len)



spam_data= data.loc[data['label']=='spam']

ham_data= data.loc[data['label']=='ham']
#Checking the length of spam and ham messages

plt.figure(figsize=(10,4))

plt.subplot(121)

sns.distplot(ham_data['length'], label='ham_length')

plt.legend()

plt.subplot(122)

sns.distplot(spam_data['length'], label= 'spam_lenth', color='orange')

plt.legend()

plt.show()
spam_data['length'].describe()
ham_data['length'].describe()
#LowerCase

msg = data['message'][0]

msg = msg.lower()



#Stopwords

from nltk.tokenize import word_tokenize

msg = word_tokenize(msg, preserve_line=False)



#Stop word removal

from nltk.corpus import stopwords

msg = [words for words in msg if words not in stopwords.words('english')]



#punctuations removal

import string

msg = " ".join(msg)

nopunc = [c for c in msg if c not in string.punctuation]

nopunc = ''.join(nopunc)



# apostrope removal



#single character removal

msg = [words for words in nopunc.split() if len(words) >1]



#Lemmatization

from nltk import stem

word_lem = stem.WordNetLemmatizer()

msg = [word_lem.lemmatize(words) for words in msg]



#Stemming

from nltk.stem import PorterStemmer

pst = PorterStemmer()

msg = [pst.stem(word) for word in msg]

msg = ' '.join(msg)

msg
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string

from nltk import stem

from nltk.stem import PorterStemmer



def text_process(msg):

    #LowerCase

    msg = msg.lower()



    #Stopwords

    msg = word_tokenize(msg, preserve_line=False)



    #Stop word removal

    msg = [words for words in msg if words not in stopwords.words('english')]



    #punctuations removal

    msg = " ".join(msg)

    nopunc = [c for c in msg if c not in string.punctuation]

    nopunc = ''.join(nopunc)



    # apostrope removal



    #single character removal

    msg = [words for words in nopunc.split() if len(words) >1]



    #Lemmatization

    word_lem = stem.WordNetLemmatizer()

    msg = [word_lem.lemmatize(words) for words in msg]



    #Stemming

    pst = PorterStemmer()

    msg = [pst.stem(word) for word in msg]

    msg = ' '.join(msg)

    

    return msg

    
#Apply text processing to each message and calculating length again

data['message']= data['message'].apply(text_process)

data['new_length'] = data['message'].apply(len)

data.head()
#After text processing lets calculate the new_length 

spam_data_new= data.loc[data['label']=='spam']

ham_data_new= data.loc[data['label']=='ham']
#Plot

plt.figure(figsize=(10,4))

plt.subplot(121)

sns.distplot(ham_data_new['new_length'], label='ham_length')

plt.legend()

plt.subplot(122)

sns.distplot(spam_data_new['new_length'], label= 'spam_lenth', color='orange')

plt.legend()

plt.show()



data.groupby(by = 'label')['new_length'].describe()
#Vectorization

from sklearn.feature_extraction.text import CountVectorizer

#Binary bag of words

count_vector = CountVectorizer(binary=True)
binary_bow= count_vector.fit(data['message'])

binary_bow.get_params
# Print total number of vocab words

print(len(binary_bow.vocabulary_))

[v for v in binary_bow.vocabulary_.items()][0:5]



### we have 7776 words in our vocabulary 
### lets take the 4th message from our message dataframe 

message4 = data['message'][3]

print(message4)

print('\n')



### use the bow_transformer and call transform function on the test message "message4"

bow4 = binary_bow.transform([message4])

print(bow4)

print('\n')

### .transform outputs the sparse matrix of indexes along with the number of times each word occurs in that index.



print(type(bow4))

print('\n')

print(bow4.ndim)

print(bow4.shape)

print('\n')



#Checking the colums are correct for words or not

print(binary_bow.get_feature_names()[1103])

print(binary_bow.get_feature_names()[2602])

print(binary_bow.get_feature_names()[2618])

print(binary_bow.get_feature_names()[3590])

print(binary_bow.get_feature_names()[5954])
#Transforming

bag_of_words= binary_bow.transform(data['message'])
### check the shape of the sparse matrix using .shape

print('Shape of Sparse Matrix: ', bag_of_words.shape)



### check the amount of non zero occurrences using .nnz

print('Amount of Non-Zero occurences: ', bag_of_words.nnz)
sparsity = (100.0 * bag_of_words.nnz / (bag_of_words.shape[0] * bag_of_words.shape[1]))



print('sparsity: {}'.format((sparsity)))

## sparsity counts the number of non zero messages vs the total number of messages
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()

tfidf_vect= tfidf_vect.fit(data['message'])
# Print total number of vocab words

print(len(tfidf_vect.vocabulary_))

[v for v in tfidf_vect.vocabulary_.items()][0:5]



### we have 7776 words in our vocabulary 
### lets take the 4th message from our message dataframe 

message4 = data['message'][3]

print(message4)

tfidf4 = tfidf_vect.transform([data['message'][3]])

print(tfidf4)

print(type(tfidf4))

print(tfidf4.ndim)

print(tfidf4.shape)
#Transforming

tfidf= tfidf_vect.transform(data['message'])
### check the shape of the sparse matrix using .shape

print('Shape of Sparse Matrix: ', tfidf.shape)



### check the amount of non zero occurrences using .nnz

print('Amount of Non-Zero occurences: ', tfidf.nnz)
sparsity = (100.0 * tfidf.nnz / (tfidf.shape[0] * tfidf.shape[1]))



print('sparsity: {}'.format((sparsity)))

## sparsity counts the number of non zero messages vs the total number of messages
from sklearn.model_selection import train_test_split
#Naive bayes for Bag of word data



#Train test split for bag_of_words data

X_train, X_test, Y_train, Y_test = train_test_split(bag_of_words, data['label'], test_size = 0.25, random_state=1)



#Building naive bayes model for BOW data

from sklearn.naive_bayes import MultinomialNB

naive_bayes_bow= MultinomialNB()

naive_bayes_bow.fit(X_train, Y_train)
#Naive bayes for Tfidf data data



#Train test split for tfidf data

x_train, x_test, y_train, y_test = train_test_split(tfidf, data['label'], test_size = 0.25, random_state=1)



#Building naive bayes model for tfidf data

naive_bayes_tfidf= MultinomialNB()

naive_bayes_tfidf.fit(x_train, y_train)
#Metrics for Bag of words model

print('The shape of X_train is {}'.format(X_train.shape))

print('The shape of X_test is {}'.format(X_test.shape))

print('\n')

print('The accuracy for Binary BOW model is {}'.format(accuracy_score(Y_test, naive_bayes_bow.predict(X_test))))

print('\n')

print('The confusion matrix for Binary BOW model is :')

print(confusion_matrix(Y_test, naive_bayes_bow.predict(X_test)))

print('\n')

print('The classification report for Binary BOW model is :')

print(classification_report(Y_test, naive_bayes_bow.predict(X_test)))
#Metrics for TFIDF Naive Bayes model

print('The shape of x_train is {}'.format(x_train.shape))

print('The shape of x_test is {}'.format(x_test.shape))

print('\n')

print('The accuracy for TFIDF Naive Bayes model is {}'.format(accuracy_score(y_test, naive_bayes_tfidf.predict(x_test))))

print('\n')

print('The confusion matrix for TFIDF Naive Bayes model is :')

print(confusion_matrix(y_test, naive_bayes_tfidf.predict(x_test)))

print('\n')

print('The classification report for TFIDF Naive Bayes model is :')

print(classification_report(y_test, naive_bayes_tfidf.predict(x_test)))
#Random Forest model for TFIDF data

from sklearn.ensemble import RandomForestClassifier

rf_tfidf = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=1)

rf_tfidf.fit(x_train, y_train)
#Metrics for TFIDF Random Forest model

print('The shape of x_train is {}'.format(x_train.shape))

print('The shape of x_test is {}'.format(x_test.shape))

print('\n')

print('The accuracy for TFIDF Random Forest model is {}'.format(accuracy_score(y_test, rf_tfidf.predict(x_test))))

print('\n')

print('The confusion matrix for TFIDF Random Forest model is :')

print(confusion_matrix(y_test, rf_tfidf.predict(x_test)))

print('\n')

print('The classification report for TFIDF Random Forest model is :')

print(classification_report(y_test, rf_tfidf.predict(x_test)))
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
msg_train, msg_test, label_train, label_test = train_test_split(data['message'], data['label'], test_size=0.3)
### observe we have summarized all the steps we did in the Pipleine which takes a list of every process we did so far

estimators =[('bow', CountVectorizer(analyzer=text_process)), ('log_reg', LogisticRegression())]

pipeline = Pipeline(estimators)
#Now we can directly pass message text data and the pipeline will do our pre-processing 

##We can treat it as a model/estimator API:

pipeline.fit(msg_train,label_train)
#Metrics for Pipeline model

print('The shape of msg_train is {}'.format(msg_train.shape))

print('The shape of msg_test is {}'.format(msg_test.shape))

print('\n')

print('The accuracy for Pipeline Log Reg model is {}'.format(accuracy_score(label_test, pipeline.predict(msg_test))))

print('\n')

print('The confusion matrix for TFIDF Random Forest model is :')

print(confusion_matrix(label_test, pipeline.predict(msg_test)))

print('\n')

print('The classification report for TFIDF Random Forest model is :')

print(classification_report(label_test, pipeline.predict(msg_test)))