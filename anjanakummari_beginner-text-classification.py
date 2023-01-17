# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import nltk

import string

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer



from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read the fake news dataset

fake_df = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")



# Assigning classses to each post

fake_df['class'] = 0 



fake_df.head()

# Read the true news dataset

true_df = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")



# Assigning classses to each post

true_df['class'] = 1



true_df.head()
# Combining both the datasets

df = pd.concat([fake_df, true_df])



# Remove irrelevant columns

df.drop(['title', 'subject', 'date'], axis=1, inplace=True)



# Drop the rows with empty Text

df["text"].dropna(inplace = True)



df.head()
'''

Text data requires preparation before you can start using it for predictive modeling. The text preprocessing steps include

but are not limited to:



1. Removing punctuation

2. Tokenizing the text into words

3. Removing stopwords

4. Lemmatizing the word tokens

'''



stopwords = nltk.corpus.stopwords.words('english')

lemmatizer = WordNetLemmatizer()



def clean_text(text):

    text_clean = "".join([char for char in text if char not in string.punctuation])

    text_clean = re.split('\W+', text.lower())

    text_clean = [word for word in text_clean if word not in stopwords]

    text_clean = " ".join([lemmatizer.lemmatize(i, 'v') for i in text_clean])

    return text_clean
df['clean_text'] = df['text'].apply(lambda x: clean_text(x))

df.head()
# Divide the dataset into training and testing data

x_train,x_test,y_train,y_test = train_test_split(df['clean_text'], df['class'], test_size=0.2, random_state=1490)



ml_pipeline = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', LogisticRegression())])



model = ml_pipeline.fit(x_train, y_train)

y_predict = model.predict(x_test)



# Evaluation Metrics

tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_predict).ravel()

specificity = (tn/(tn+fp))*100

print("Accuracy: {}%".format(round(metrics.accuracy_score(y_test, y_predict)*100,2)))

print("Sensitivity: {0:0.2f}%".format(metrics.recall_score(y_test, y_predict)*100))

print("Specificity: {0:0.2f}%".format(specificity))

print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_predict))
ml_pipeline = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', LinearSVC())])



model = ml_pipeline.fit(x_train, y_train)

y_predict = model.predict(x_test)



# Evaluation Metrics

tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_predict).ravel()

specificity = (tn/(tn+fp))*100

print("Accuracy: {}%".format(round(metrics.accuracy_score(y_test, y_predict)*100,2)))

print("Sensitivity: {0:0.2f}%".format(metrics.recall_score(y_test, y_predict)*100))

print("Specificity: {0:0.2f}%".format(specificity))

print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_predict))
ml_pipeline = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', RandomForestClassifier())])



model = ml_pipeline.fit(x_train, y_train)

y_predict = model.predict(x_test)



# Evaluation Metrics

tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_predict).ravel()

specificity = (tn/(tn+fp))*100

print("Accuracy: {}%".format(round(metrics.accuracy_score(y_test, y_predict)*100,2)))

print("Sensitivity: {0:0.2f}%".format(metrics.recall_score(y_test, y_predict)*100))

print("Specificity: {0:0.2f}%".format(specificity))

print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_predict))
from sklearn.metrics import precision_recall_fscore_support as score



def train_RF(n_est, depth):

        rf = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', RandomForestClassifier(n_estimators = n_est, max_depth = depth, n_jobs = -1))])

        rf_model = rf.fit(x_train, y_train)

        y_pred = rf_model.predict(x_test)

        precision, recall, fscore, support = score(y_test, y_pred, average = 'binary')

        print('Est: {} / Depth: {} ------ Precision: {} / Recall: {} / Accuracy: {}'.format(n_est, depth, round(precision, 3),

                                                                                           round(recall, 3), 

                                                                                        round((y_pred == y_test).sum()/len(y_pred), 3)))



'''

We are going to run the model for various number of estimators with increasing depth values.

'''

for n_est in [10, 50, 100]:

    for depth in [10, 20, 30, None]:

        train_RF(n_est, depth)