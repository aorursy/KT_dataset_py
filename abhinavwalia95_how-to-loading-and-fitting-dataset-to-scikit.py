import json

import pandas as pd

import numpy as np

from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import Perceptron

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support, f1_score
dframe = pd.read_csv("../input/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
dframe.dropna(inplace=True)
dframe[dframe.isnull().any(axis=1)].size
dframe =  dframe[:5000]

x_df = dframe.drop(['Unnamed: 0', 'sentence_idx', 'tag'], axis=1)

x_df.head()
vectorizer = DictVectorizer(sparse=False)

x = vectorizer.fit_transform(x_df.to_dict("records"))

x.shape
y = dframe.tag.values

all_classes = np.unique(y)

all_classes.shape

y.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(x_train.shape)

print(y_train.shape)
clf = Perceptron(verbose=10, n_jobs=-1, n_iter=5)

all_classes = list(set(y))

clf.partial_fit(x_train, y_train, all_classes)
print(f1_score(clf.predict(x_test), y_test, average="micro"))