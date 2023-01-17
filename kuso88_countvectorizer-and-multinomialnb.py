# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

training_set = pd.read_csv('../input/labeledTrainData.tsv', sep='\t')

y_train = training_set['sentiment'].values

x_train = training_set['review'].values



vec = CountVectorizer()

X = vec.fit_transform(x_train)



nb = MultinomialNB()

nb.fit(X, y_train)



test_data = pd.read_csv('../input/testData.tsv', sep='\t')

test_reviews = test_data['review'].values



X = vec.transform(test_reviews)

predictions = nb.predict(X)



data = {'id': test_data['id'].values, 'sentiment':predictions}

df = pd.DataFrame(data=data)

df.to_csv('./predictions.csv', index=False)

df.head()
