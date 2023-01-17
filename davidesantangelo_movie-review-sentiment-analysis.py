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
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import accuracy_score
df = pd.read_csv('../input/movie-review/movie_review.csv')
X = df['text']

y = df['tag']
vect = CountVectorizer(ngram_range=(1, 2))



X = vect.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = BernoulliNB()



model.fit(X_train, y_train)



p_train = model.predict(X_train)

p_test = model.predict(X_test)
acc_train = accuracy_score(y_train, p_train)

acc_test = accuracy_score(y_test, p_test)



print(f'Train ACC: {acc_train}, Test ACC: {acc_test}')