import numpy as np # linear algebra

import pandas as pd # data processing/manipulation

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LinearRegression



from collections import Counter

import nltk

import seaborn as sns

import string

from nltk.corpus import stopwords



import os

print(os.listdir("../input"))

#Importing Data

data = pd.read_csv('../input/consumer-reviews-of-amazon-products/1429_1.csv')

data.head()
#34660 columns and 21 columns.

data.shape
#Checking missing values.

data.isnull().sum()
review=pd.DataFrame(data.groupby('reviews.rating').size().sort_values(ascending=False).rename('No of Users').reset_index())

review.head()
#Though the four columns below contains very less missing values, it would do no harm in removing NA values.

data1 = data[['reviews.rating' , 'reviews.text' , 'reviews.title' , 'reviews.username']]

final = data1.dropna()

final.head()
#No NA values

fact =  final[final["reviews.text"].isnull()]

fact.head()
rating = final[(final['reviews.rating'] == 1) | (final['reviews.rating'] == 5)]

rating.shape
y = rating['reviews.rating']

x = rating['reviews.text'].reset_index()
len(y)
X = x['reviews.text']

print(X)
print(len(X))
import nltk

from nltk.corpus import stopwords

set(stopwords.words('english'))
import nltk

nltk.download('punkt')
import nltk

nltk.download('wordnet')
#Lemmatization

from nltk.stem import WordNetLemmatizer 

from nltk.tokenize import word_tokenize 

lemmatizer = WordNetLemmatizer() 

# lemmatize string 

def lemmatize_word(text): 

    word_tokens = word_tokenize(text) 

    # provide context i.e. part-of-speech 

    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens] 

    return lemmas 

  

text = 'Inexpensive tablet for him to use and learn on, step up from the NABI. He was thrilled with it, learn how to Skype on it already....'

lemmatize_word(text) 
import string

from nltk.corpus import stopwords

# stop=set(stopwords.words('english'))

def text_process(text):

  

    nopunc = [char for char in text if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
import nltk

nltk.download('stopwords') #To downlaod 'stopwords' in your notebook

sample_text = ("Inexpensive tablet for him to use and learn on, step up from the NABI. He was thrilled with it, learn how to Skype on it already....")

print(text_process(sample_text))
from sklearn.feature_extraction.text import CountVectorizer



transformer = CountVectorizer(analyzer=text_process).fit(X)
#The numbers are not count, they are position in sparse vector.

transformer.vocabulary_
print(transformer)
len(transformer.vocabulary_)
review = X[24]

bow = transformer.transform([review])

bow
print(bow)
X = transformer.transform(X)
#Lets start training the model

from sklearn.model_selection import train_test_split

#using 30% of the data for testing, this will be revised once we do not get the desired accuracy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.naive_bayes import MultinomialNB

naive = MultinomialNB()

naive.fit(X_train, y_train)
pred = naive.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred))

print('\n')



print(classification_report(y_test, pred))

naive.score(X_train, y_train)
from sklearn.svm import SVC

clf = SVC()

clf.fit(X_train, y_train) 

svm = clf.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, pred))

print('\n')

print(classification_report(y_test, svm))

svm = clf.predict(X_test)

clf.score(X_train,y_train)
#Both the models relatively provides better accuracy.

methods = ["Multinomial Naive Bayes","SVM"]

accuracy = [0.9839,0.9908]

sns.set_style("whitegrid")

plt.figure(figsize=(6,8))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %",fontsize=14)

plt.xlabel("ML Models",fontsize=14)

sns.barplot(x=methods, y=accuracy)

plt.show()