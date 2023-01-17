# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import sqlite3

import numpy as np

import pandas as pd

import sklearn

#from sklearn.model_selection import train_test_split # ended up not using this today

import string

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import linear_model

con = sqlite3.connect("../input/database.sqlite")
tiny = pd.read_sql("select * from Reviews limit 2", con)

tiny
tiny = pd.read_sql("select Id, ProductId, UserId, Score, Summary from Reviews limit 2", con)

tiny.head()
translator = str.maketrans({key: None for key in string.punctuation})

# this line was a source of immense python 3 pain

def remove_punctuation(text):

    return text.translate(translator).lower()
tiny['Summary_clean1'] = tiny['Summary'].apply(remove_punctuation)
tiny_count_vect = CountVectorizer()

tiny_matrix = tiny_count_vect.fit_transform(tiny['Summary_clean1'])
print(tiny_count_vect.vocabulary_)

print()

print(tiny_matrix)

print()

tiny.head()
tiny_model = linear_model.LogisticRegression() # create a model with default parameters

tiny_model.fit(tiny_matrix, tiny['Score']) # feed it the data and the scores



# and have a look at what we've got

print(tiny_model.coef_)
prediction_row_1 = np.dot(tiny_model.coef_, [0,0,1,1,1,0,1])

print(prediction_row_1)

prediction_row_2 = np.dot(tiny_model.coef_, [1,1,0,0,0,1,0])

print(prediction_row_2)
tiny_model.predict(tiny_count_vect.transform(tiny['Summary_clean1']))
tiny_model.predict_proba(tiny_count_vect.transform(tiny['Summary_clean1']))
import math

print(math.tanh(1.05409158))

print(math.tanh(-0.86962972))
small = pd.read_sql("select Id, ProductId, UserId, Score, Summary, Text from Reviews limit 1004", con) # get data

small['Summary_clean1'] = small['Summary'].apply(remove_punctuation) # clean it

small_train = small[:1000]

small_test = small[-4:]

small_count_vect = CountVectorizer()

small_matrix = small_count_vect.fit_transform(small_train['Summary_clean1']) # prepare training data

small_model = linear_model.LogisticRegression() # instantiate the model

small_model.fit(small_matrix, small_train['Score']) # train the model on training data



# see how the model goes on the four rows of test data

print('category predictions\n', small_model.predict(small_count_vect.transform(small_test['Summary_clean1'])))

print()

print('probability predictions\n', small_model.predict_proba(small_count_vect.transform(small_test['Summary_clean1'])))

print()

print('actual scores\n', small_test['Score'])