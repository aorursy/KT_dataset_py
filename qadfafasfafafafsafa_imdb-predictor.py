# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf

from tensorflow import keras

from sklearn.feature_extraction.text import *

%matplotlib inline



# Any results you write to the current directory are saved as output.
X_train = pd.read_csv('../input/train.csv', index_col = 'Id')

X_test = pd.read_csv('../input/test.csv', index_col = 'Id')
# count = CountVectorizer()

tf = TfidfVectorizer(ngram_range = (1, 2), stop_words = 'english')



# count.fit(X_train.review)

tf.fit_transform(X_train.review)





# X_train_counts = count.transform(X_train.review)

X_train_tf = tf.transform(X_train.review)



# X_test_counts = count.transform(X_test.review)

X_test_tf = tf.transform(X_test.review)
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(alpha = 3)

clf.fit(X_train_tf, X_train.y)
y_pred = clf.predict(X_test_tf)

pred = pd.DataFrame({'y': y_pred}, index = X_test.index)

pred.to_csv('prediction.csv')