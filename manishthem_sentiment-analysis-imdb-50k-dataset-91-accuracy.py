# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filepath = os.path.join(dirname, filename) 
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(filepath)
df.sentiment.value_counts()
df.info()
df.head()
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
text_count_matrix = tfidf.fit_transform(df.review)
#splitting the complete dataset in test and training dataset:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(text_count_matrix, df.sentiment, test_size=0.30, random_state=2)
#converting the sentiments (positive and negatives) to 1 and 0. 
y_train = (y_train.replace({'positive': 1, 'negative': 0})).values
y_test = (y_test.replace({'positive': 1, 'negative': 0})).values
# let's use Naive Bayes classifier and fit our model:
from sklearn.naive_bayes import MultinomialNB 
MNB = MultinomialNB()
MNB.fit(x_train, y_train)
#4. Evaluating the model
from sklearn import metrics
accuracy_score = metrics.accuracy_score(MNB.predict(x_test), y_test)
print("accuracy_score without data pre-processing = " + str('{:04.2f}'.format(accuracy_score*100))+" %")
#let's investigate what kind of special characters and language is used by the reviewers to review the content. 
#we can observe some html tags
#use of parenthesis
#punctuation (apostrophy, '' e.t.c)

import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()
#print(df.review[4])

processed_review = []
single_review = "string to iniialize <br /> my email id is charilie@waoow.com. You can also reach to me at charlie's "
reviews = df.review
for review in range(0,50000):
    single_review = df.loc[review,'review']
    
    #start processing the single_review 
    
    #removing html tags:
    single_review = re.sub('<.*?>',' ',single_review)
    #removing special characters (punctuation) '@,!' e.t.c.
    single_review = re.sub('\W',' ',single_review)
    #removing single characters
    single_review = re.sub('\s+[a-zA-Z]\s+',' ', single_review)
    #substituting multiple spaces with single space
    single_review = re.sub('\s+',' ', single_review)
   
    #removing stop words
    #word_tokens = []
    word_tokens = word_tokenize(single_review)
    #lemmatization
    #lemmatized_sentence = " ".join(lemmatizer.lemmatize(token) for token in word_tokens if token not in stop_words)
    filtered_sentence = []
    #filtered_sentence.append([w for w in word_tokens if w not in stop_words])
    filtered_sentence2 = " ".join([w for w in word_tokens if w not in stop_words])
    
    
    #compile all the sentences to make a complete dictionary of processed reviews
    processed_review.append(filtered_sentence2)
    
print(processed_review[10])
#print(filtered_sentence2)
text_count_matrix2 = tfidf.fit_transform(processed_review)
X_train, X_test, Y_train, Y_test = train_test_split(text_count_matrix2, df.sentiment, test_size=0.30, random_state=2)
Y_train = (Y_train.replace({'positive': 1, 'negative': 0})).values
Y_test = (Y_test.replace({'positive': 1, 'negative': 0})).values
MNB.fit(X_train, Y_train)
#4. Evaluating the model
accuracy_score = metrics.accuracy_score(MNB.predict(X_test), Y_test)
print(str('{:04.2f}'.format(accuracy_score*100))+" %")
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report: \n", classification_report(Y_test, MNB.predict(X_test),target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(Y_test, MNB.predict(X_test)))
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
LSVC = LinearSVC()
LSVC.fit(X_train, Y_train)
accuracy_score = metrics.accuracy_score(LSVC.predict(X_test), Y_test)
print("Linear SVC accuracy = " + str('{:04.2f}'.format(accuracy_score*100))+" %")
print("Classification Report: \n", classification_report(Y_test, LSVC.predict(X_test),target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(Y_test, LSVC.predict(X_test)))
SGDC = SGDClassifier()
SGDC.fit(X_train, Y_train)
predict = SGDC.predict(X_test)
accuracy_score = metrics.accuracy_score(predict, Y_test)
print("Stocastic Gradient Classifier accuracy = " + str('{:04.2f}'.format(accuracy_score*100))+" %")
print("Classification Report: \n", classification_report(Y_test, predict,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(Y_test, predict))
LR = LogisticRegression()
LR.fit(X_train, Y_train)
predict = LR.predict(X_test)
accuracy_score = metrics.accuracy_score(predict, Y_test)
print("LR = " + str('{:04.2f}'.format(accuracy_score*100))+" %")
print("Classification Report: \n", classification_report(Y_test, predict,target_names=['Negative','Positive']))
print("Confusion Matrix: \n", confusion_matrix(Y_test, predict))
