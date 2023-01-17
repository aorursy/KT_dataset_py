import numpy as np

import pandas as pd

import random

import matplotlib.pyplot as plt

import seaborn as sns

import re 

from pandasql import sqldf 

import datetime
test_df = pd.read_csv('../input/chow-sentiment/full_reviews.csv')
# we now delete the "neutral" ratings ex: 3

not_neutral = test_df[test_df.stars !=3]
null_count = not_neutral.isnull().sum()

null_count
not_neutral['binary_class']  = np.where(not_neutral['stars'] > 3, 1, 0)
not_neutral
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(not_neutral['one_liner'], not_neutral['binary_class'], random_state = 0)

print(' ')

print('X_train shape: ' + str(X_train.shape))
from sklearn.feature_extraction.text import CountVectorizer

#creating variable which assigns X_train to numbers

vect = CountVectorizer().fit(X_train)

#translates numbers back to text

vect.get_feature_names()[1:10]
#length of total words

len(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)

print (X_train_vectorized.toarray())
#creating log regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_vectorized, y_train)
#calculating AUC

from sklearn.metrics import roc_auc_score

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))
#creating array variable of all the words

feature_names = np.array(vect.get_feature_names())

#creating array of all the regression coefficients per word

coef_index = model.coef_[0]

#creating df with both arrays in it

df = pd.DataFrame({'Word':feature_names, 'Coef': coef_index})

#sorting by coefficient

sorted_df = df.sort_values('Coef')
sorted_df 
good_words = sorted_df.tail(10)

worst_words = sorted_df.head(10)
import plotly.express as px 

import plotly.graph_objects as go
fig = px.bar(good_words, x='Word', y='Coef', color="Coef", title="Most Positively Correlated Words")

fig.show()
fig = px.bar(worst_words, x='Word', y='Coef', color="Coef", title="Doordash Most Negatively Correlated Words")

fig.show()