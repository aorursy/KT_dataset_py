import pandas as pd

import seaborn as sns

import numpy as np

import time

import re

import nltk

import math

import os

import matplotlib.pyplot as plt

import sklearn.metrics



from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.linear_model import RidgeClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn import linear_model

from sklearn.linear_model import Perceptron



from sklearn import svm

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn import linear_model

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import PassiveAggressiveClassifier



from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import roc_auc_score

from Bio.SeqUtils.ProtParam import ProteinAnalysis

import sklearn.metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



import pickle



%matplotlib inline

proteins = pd.read_csv('../input/protein-locations/protein-SevenLocations-Feb12.csv')



proteins.head()
proteins.shape
# permute/scramble/shuffle rows of the dataframe

proteins = proteins.sample(frac=1)
proteins.location.value_counts()
proteins.label.value_counts()
# remove the duplicate protein sequences

proteins = proteins.drop_duplicates(subset='sequence', keep="first")



# remove nan from 'sequence' column

proteins = proteins[proteins['sequence'].notnull()]



proteins.shape
proteins.label.value_counts()
# Peptide count is used for analysis

peptide_size = 5

vect = CountVectorizer(min_df=1,token_pattern=r'\w{1}',ngram_range=(peptide_size,peptide_size))
X = vect.fit_transform(proteins.sequence)

y = proteins.location
# Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state =42)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
y_test.value_counts()
%%time

mnb = MultinomialNB()

mnb.fit(X_train, y_train)

# evaluate accuracy of our model on test data

print("MNB classifier Score: {:.2f}".format(mnb.score(X_test, y_test)))

print()
%%time

pac = PassiveAggressiveClassifier()

pac.fit(X_train, y_train)

# evaluate accuracy of our model on test data

print("Passive Aggressive classifier Score: {:.2f}".format(pac.score(X_test, y_test)))

print()
vote_prediction_pac = pac.predict(X_test)

print(classification_report(vote_prediction_pac, y_test))
%%time

pac2 = PassiveAggressiveClassifier(loss='squared_hinge')

pac2.fit(X_train, y_train)

# evaluate accuracy of our model on test data

print("Passive Aggressive classifier with squared hinge loss Score: {:.2f}".format(pac2.score(X_test, y_test)))

print()
# Generate Confusion Matrix 

actual = y_test

predictions = pac2.predict(X_test)

print('Confusion Matrix for Majority Vote Model')

print()

cm = confusion_matrix(actual,predictions)

print(cm)
%%time

sgd = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

sgd.fit(X_train, y_train)

# evaluate accuracy of our model on test data

print("SGD classifier Score: {:.2f}".format(sgd.score(X_test, y_test)))

print()
from sklearn.model_selection import cross_val_score, cross_val_predict
pac_cv = PassiveAggressiveClassifier()

scores = cross_val_score(pac_cv,X,y, cv = 5)

print("Cross-validation scores for Passive Aggressive Classifier: {}".format(scores))

print()

print("The average accuracy score for Passive Aggressive Classifier is: ")

print(np.mean(scores))