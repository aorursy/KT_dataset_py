

# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Importing the dataset

dataset = pd.read_csv('../input/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Check the dataset!

dataset.head()
#check the null values!

dataset.isnull().sum()


# Cleaning the texts

import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = [] #Creating an empty list

for i in range(0, 1000):

    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])#Remove all specail character like('',:,;,%,*,@,$,!,#) 

    #Convert complete sentences into lowercase. 

    review = review.lower() 

    #Spliting each sentences into different words.

    review = review.split()

    #Stemming example is love-loving,loved,lovely.etc.

    ps = PorterStemmer()

    #Removing unnecessary words like is,of,for,this...and only focusing on main meaning of the sentence. 

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)
print(corpus)
print(review)


# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1].values

print(X) #Separting every unique value and mapping it as 1 just like encode and decode..
print(y) # liked or not!
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
Accuracy=(55+91)/200

Accuracy
Error_rate=(42+12)/200

Error_rate