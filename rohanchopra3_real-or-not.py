# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing, naive_bayes, svm

from sklearn.feature_extraction.text import TfidfVectorizer ,TfidfTransformer

from sklearn.metrics import accuracy_score

from string import punctuation

from nltk.corpus import stopwords

from nltk import word_tokenize

from sklearn.feature_selection import SelectPercentile, f_classif



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
training_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

testing_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

corpus = []

for values in training_data['text']:

    corpus.append(values)

    

stop_words = stopwords.words('english') + list(punctuation)



vectorizer = TfidfVectorizer(use_idf=True, stop_words=stop_words,smooth_idf=True)

Train = vectorizer.fit_transform(corpus)

test = vectorizer.transform(testing_data['text'])

clf = linear_model.RidgeClassifier()

scores = model_selection.cross_val_score(clf, Train, training_data["target"], cv=3, scoring="f1")

scores

clf.fit(Train, training_data["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(test)

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)