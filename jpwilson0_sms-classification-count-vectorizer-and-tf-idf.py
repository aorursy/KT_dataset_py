import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))



email_data = pd.read_csv('../input/spam.csv', encoding = 'latin-1' )
email_data.head(20)
email_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis  = 1, inplace = True)
email_data.head(10)
email_data = email_data.rename(columns = {"v1": "label", "v2": "text"})
email_data.head()
sns.countplot(email_data['label'], label = "Count of the Labels")
email_data['length'] = email_data['text'].apply(len)
email_data.head()
email_data['length'].hist(bins = 50, color = 'g')
#Let's import the needed libraries

import string

from nltk.corpus import stopwords
def clean_review(review):

    remove_punctuation = [word for word in review if word not in string.punctuation]

    join_characters = ''.join(remove_punctuation)

    remove_stopwords = [word for word in join_characters.split() if word.lower() not in stopwords.words('english')]

    cleaned_review = remove_stopwords

    return cleaned_review
#The original data

email_data['text'][4]
#Applying the cleaning function

clean_review(email_data['text'][4])
from sklearn.feature_extraction.text import CountVectorizer

#Define the clean_review function as an argument of the CountVectorizer.

count_vectorizer = CountVectorizer(analyzer = clean_review)

#Fit the countvectorizer to the dataset

email_countvec = count_vectorizer.fit_transform(email_data['text'])
print(count_vectorizer.get_feature_names())
email_countvec.shape
#Let's have a quick look at the data once again

email_data.head()
#Using LabelEncoder from sklearn

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

email_data['label'] = le.fit_transform(email_data['label'])
#Assign X and y

X = email_countvec

y = email_data['label']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
from sklearn.naive_bayes import MultinomialNB



classifier = MultinomialNB() #Using the default parameters

classifier.fit(X_train, y_train)
#Making predictions to the Test Set

y_predictions = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report



cm = confusion_matrix(y_test, y_predictions)



#Taking a look at the Confusion Matrix with a Heatmap

sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predictions))
from sklearn.feature_extraction.text import TfidfVectorizer

tfid = TfidfVectorizer()

tfidvec = tfid.fit_transform(email_data['text'])
tfidvec.shape
print(tfid.get_feature_names())
print(tfidvec[:,:])
#Assigning X2 and y2

X2 = tfidvec

y2 = email_data['label']
from sklearn.model_selection import train_test_split

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.25)
from sklearn.naive_bayes import MultinomialNB



classifier2 = MultinomialNB() #Using the default parameters

classifier2.fit(X_train2, y_train2)
#Making predictions to the Test Set

y_predictions2 = classifier2.predict(X_test2)
from sklearn.metrics import confusion_matrix, classification_report



cm2 = confusion_matrix(y_test2, y_predictions2)



#Taking a look at the Confusion Matrix with a Heatmap

sns.heatmap(cm2, annot=True)
print(classification_report(y_test2, y_predictions2))