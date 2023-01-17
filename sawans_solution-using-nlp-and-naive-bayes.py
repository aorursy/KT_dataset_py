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
#Loading the data

train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



print("Total number of records in training set: ",len(train_df))

print("Total number of records in test set: ",len(test_df))
#Analzing the data

print("Number of disaster tweets: ",len(train_df[train_df['target']==1]))

print("Number of non-disaster tweets: ",len(train_df[train_df['target']==0]))
#Importing required libraries

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

import re


#Function to clean text

def clean_text(text):

    temp = text.lower()

    temp = re.sub('[^a-zA-Z]',' ',temp)

    temp = re.sub('\n', " ", temp)

    temp = re.sub('\'', "", temp)

    temp = re.sub('-', " ", temp)

    temp = re.sub(r"(http|https|pic.)\S+"," ",temp)

    

    return temp
lemmatizer = WordNetLemmatizer()

stopWords = stopwords.words('english')

stopWords.remove('no')

stopWords.remove('not')



#Function to lemmatize words and remvoing stopwords

def lemmatize_removestopwords(text):

    temp=''

    tokenized_words = word_tokenize(text)

    temp = [lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='v'),pos='a')

                 for word in tokenized_words\

                 if word not in stopWords]

    temp =' '.join(temp)

    return temp
#Example of applying lemmatization and stop words removal

text1 = 'He is a better human being. I am going to see him'

text2 = lemmatize_removestopwords(text1)

text2

#Saving the clean and lemmatized text in a new column called 'clean

train_df['clean'] = train_df['text'].apply(clean_text)

train_df['clean'] = train_df['clean'].apply(lemmatize_removestopwords)

train_df['clean']
#Splitting training data into train and validation set

X = train_df['clean']

y = train_df['target']



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25,random_state=42) 
#Vectorizing the text using TFIDF

vectorizer = TfidfVectorizer()

X_train_vect = vectorizer.fit_transform(X_train)

X_val_vect = vectorizer.transform(X_val)

#Note in the above statement, we only use fit_tranform with training data and just tranform 

#on validation data. fit_tranform also calculates the number of tokens in the data. Hence we use just

#tranform on validation data to ensure the number of token are same



#Verify this by printing the shape of both vectors

print("Number of columns in training vector: ",X_train_vect.shape[1])

print("Number of columns in validation vector: ",X_val_vect.shape[1])

#Bulding the classifier

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(X_train_vect,y_train)
#Checking the accuracy score

y_pred = classifier.predict(X_val_vect)



from sklearn.metrics import accuracy_score

accuracy_score(y_val,y_pred)
#Experimenting with another classifier

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier()



#Using GridSearchCV to find best hyperparameters for the model

from sklearn.model_selection import GridSearchCV

paramGrid = {'min_samples_leaf':[5,10,15,20],'n_estimators':[50,100,150,200],

             'criterion':['gini','entropy']}

gridSearchCV = GridSearchCV(rf_classifier,param_grid=paramGrid)

gridSearchCV.fit(X_train_vect,y_train)

gridSearchCV.best_params_
#Creating the final model using best params

rf_classifier_final = RandomForestClassifier(criterion='entropy',min_samples_leaf=5,n_estimators=200)

rf_classifier_final.fit(X_train_vect,y_train)
#Checking the accuracy score

y_pred_2 = rf_classifier_final.predict(X_val_vect)



accuracy_score(y_val,y_pred_2)
#Confusion matrix for the naive bayes classifer

from sklearn.metrics import confusion_matrix

confusion_matrix(y_val,y_pred)



#Vlaues in the diagonal indicate True positive and false negatives. More number on the diagonal means

#better accuracy at prediction
#Creating submission file

#We will be using the entire train data for training again without splitting



#Load data

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
#Preprocessing



### apply preprocessing on train data

train['clean'] = train['text'].apply(clean_text)

train['clean'] = train['clean'].apply(lemmatize_removestopwords)



### apply preprocessing on test data

test['clean'] = test['text'].apply(clean_text)

test['clean'] = test['clean'].apply(lemmatize_removestopwords)



X_train = train['clean']

y_train = train['target']

X_test = test['clean']
#Text Vectorization using TF-IDF

vectorizer = TfidfVectorizer()



X_train_vect = vectorizer.fit_transform(X_train)

X_test_vect = vectorizer.transform(X_test)
#Creating classifier and predicting results

clf = MultinomialNB()

clf.fit(X_train_vect, y_train)



y_pred = clf.predict(X_test_vect)
#Saving the results to csv file for submission

result = pd.DataFrame({'id':test['id'], 'target':y_pred})

result.to_csv('mnb_disaster_tweets.csv', index=False)