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

root = '/kaggle/input/nlp-getting-started/'

# Any results you write to the current directory are saved as output.
from sklearn import feature_extraction, linear_model, model_selection, preprocessing



train_df = pd.read_csv(os.path.join(root, 'train.csv'))

test_df = pd.read_csv(os.path.join(root, 'test.csv'))





train_df.head(10)



count_vectorizer = feature_extraction.text.CountVectorizer(max_df = 159, stop_words = 'english')

train_vectors = count_vectorizer.fit_transform(train_df["text"])

train_vectors[0].todense().shape

test_vectors = count_vectorizer.transform(test_df['text'])

test_vectors[0].todense().shape

clf = linear_model.RidgeClassifier(alpha = 20)

clf.fit(train_vectors, train_df['target'])

scores = model_selection.cross_val_score(clf, train_vectors, train_df['target'], cv=3, scoring = 'f1')

scores
test_pred = clf.predict(test_vectors)

ans = pd.DataFrame(test_pred, test_df['id'], columns = {'target'})

ans.to_csv('Disaster_pred.csv')