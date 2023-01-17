import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# import all the required libraries

import pandas as pd

import numpy as np 

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
# dataset courtesy Rishabh Misra & Kaggle

# Link: https://www.kaggle.com/rmisra/news-category-dataset

data = pd.read_json("/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json", lines=True)
data.head()
# Let's see how many categories we have here

print(f"Total unique categories are: {len(data['category'].value_counts())}")

print(f"Count of occurance of each category:")

data['category'].value_counts()
#check for Null Data

data.isnull().sum()
# Check of spaces in column headline - using enumerate

spaces = []

for i, x in enumerate(data['headline']):

    if type(x) == str:

        if x.isspace():

            spaces.append(i)

        

print(len(spaces), 'spaces in index: ', spaces)
# Check of spaces in column short desc - using itertuples

blanks = []  # start with an empty list



for i,cat,hl,au,l,sd,dt in data.itertuples():  # iterate over the DataFrame

    if type(sd)==str:            # avoid NaN values

        if sd.isspace():         # test 'review' for whitespace

            blanks.append(i)     # add matching index numbers to the list

        

print(len(blanks), 'blanks: ', blanks)
# Since the goal of this exercise if to identify category based on headline and short description, 

# we choose to merge them, as the vectorizer functions can't process multiple columns

X = data['headline']+data['short_description']

y = data['category']
X.head()
# Split the data into 70-30 i.e. test size of 30% to check the accuracy of the training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77)



#Let's check the shape of the splitted data

print(f"Training Data Shape: {X_train.shape}")

print(f"Testing Data Shape: {X_test.shape}")
# Let's first try with Count Vectorizer from scikit learn

cv = CountVectorizer()



X_train_cv = cv.fit_transform(X_train)

X_train_cv.shape
from sklearn.svm import LinearSVC

clf = LinearSVC()

clf.fit(X_train_cv,y_train)
# Let's test it for the first 2 articles in the Test dataset

X_test1 = X_test[0:2]

print(X_test1)
X_test1_cv = cv.transform(X_test1)

clf.predict(X_test1_cv)
# Transform the test data before predicting

X_test_cv = cv.transform(X_test)
# Form a prediction set

predictions = clf.predict(X_test_cv)
# Report the confusion matrix

print(metrics.confusion_matrix(y_test,predictions))

# Print a classification report

print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
# single command to create a pipeline of activities...vectorize and classify the text, in this case

clf_cvec_lsvc = Pipeline([('cvec', CountVectorizer()),

                     ('clf', LinearSVC())])



# Feed the training data through the pipeline

clf_cvec_lsvc.fit(X_train, y_train)
# Form a prediction set

# No need to convert the test data. Classifier cretaed in the pipeline will take care of it

predictions = clf_cvec_lsvc.predict(X_test)

# Report the confusion matrix

print(metrics.confusion_matrix(y_test,predictions))

# Print a classification report

print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
clf_tfidf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),

                     ('clf', LinearSVC())])



# Feed the training data through the pipeline

clf_tfidf_lsvc.fit(X_train, y_train)
# Form a prediction set

predictions = clf_tfidf_lsvc.predict(X_test)

# Print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
clf_tfidf_mnb = Pipeline([('tfidf', TfidfVectorizer()),

                     ('clf', MultinomialNB())])



# Feed the training data through the pipeline

clf_tfidf_mnb.fit(X_train, y_train)  
# Form a prediction set

predictions = clf_tfidf_mnb.predict(X_test)

# Print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
clf_tfidf_lr = Pipeline([('tfidf', TfidfVectorizer()),

                     ('clf', LogisticRegression())])



# Feed the training data through the pipeline

clf_tfidf_lr.fit(X_train, y_train)
predictions = clf_tfidf_lr.predict(X_test)

# Print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
clf_tfidf_knc = Pipeline([('tfidf', TfidfVectorizer()),

                     ('clf', KNeighborsClassifier())])



# Feed the training data through the pipeline

clf_tfidf_knc.fit(X_train, y_train)
predictions = clf_tfidf_knc.predict(X_test)

# Print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
clf_tfidf_rfc = Pipeline([('tfidf', TfidfVectorizer()),

                     ('clf', RandomForestClassifier())])



# Feed the training data through the pipeline

clf_tfidf_rfc.fit(X_train, y_train)
predictions = clf_tfidf_rfc.predict(X_test)

# Print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))
# Create list of StopWords

import nltk

from nltk.corpus import stopwords

stopwords = stopwords.words('english')

print(stopwords)
clf_tfidf_lsvc2 = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopwords)),

                     ('clf', LinearSVC())])



# Feed the training data through the pipeline

clf_tfidf_lsvc2.fit(X_train, y_train)
predictions = clf_tfidf_lsvc2.predict(X_test)

# Print the overall accuracy

print(metrics.accuracy_score(y_test,predictions))