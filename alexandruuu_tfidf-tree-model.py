# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import spacy
from tensorflow import keras
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
data = pd.read_csv("../input/train.csv",header=None,low_memory=False)
data_test = pd.read_csv("../input/test.csv",header=None,low_memory=False)
sentences = data[1][1:]
sentences_test = data_test[1][1:]
Y = data[2][1:]
Y = Y.values
tfidf_vec = TfidfVectorizer(min_df = 5)
X = tfidf_vec.fit_transform(list(sentences))
tree_model = DecisionTreeClassifier().fit(X, Y)
X_test = tfidf_vec.transform(list(sentences_test))

preds = tree_model.predict(X_test)
preds = [int(x) for x in preds]
sub = pd.read_csv("../input/sample_submission.csv")
sub["prediction"] = preds
sub.to_csv("submission.csv",index=False)

#import os
#print(os.listdir("../input"))