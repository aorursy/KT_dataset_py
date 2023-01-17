# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/amazon-alexa-reviews-sentiment-analysis/amazon_alexa1.tsv',sep='\t')
data.head()

data.isnull().sum()
##Lets Import the required lib for our analysis

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import re
##Data Preprocessing

lemmatizer=WordNetLemmatizer()

corpus=[]

for i in range(0,3150):

    review=re.sub('[^a-zA-Z]',' ',data['verified_reviews'][i])

    review=review.lower()

    review=review.split()

    review=[lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]

    review=' '.join(review)

    corpus.append(review)
#Creating a Bag of Words



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer()

X=tfidf.fit_transform(corpus).toarray()

y=data.iloc[:,4].values

#Splitting the data into train and test dataset



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)



#Fitting the model(In my case I am using NaiveBayes)

from sklearn.naive_bayes import MultinomialNB

classifier=MultinomialNB()

classifier.fit(X_train,y_train)
##Predicting the values



y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cm=confusion_matrix(y_test,y_pred)

accuracy=accuracy_score(y_test, y_pred)

cr=classification_report(y_test, y_pred)

print('Confusion Matrix:\t \n',cm)

print('Accuracy: ', accuracy)

print(cr)


##Lets check using Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

classifier.fit(X_train,y_train)



##Predicting the values



y_pred=classifier.predict(X_test)



from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cm=confusion_matrix(y_test,y_pred)

accuracy=accuracy_score(y_test, y_pred)

cr=classification_report(y_test, y_pred)

print('Confusion Matrix:\t \n',cm)

print('Accuracy: ', accuracy)

print(cr)
