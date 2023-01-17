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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv("../input/train.csv",header=None,low_memory=False)
data_test = pd.read_csv("../input/test.csv",header=None,low_memory=False)
sentences = data[1][1:]
sentences_test = data_test[1][1:]
X = sentences.values
Y = data[2][1:]
Y = Y.values
glove_model = spacy.load('en_core_web_lg')
glove_model.remove_pipe('ner')

Y = data[2][1:]
Y = Y.values
sentences = data[1][1:]




insincere = X[Y == '1']
sincere = X [Y == '0']
a = []
y = []
for sent in insincere:
    a.append(glove_model(sent).vector)
    y.append(1)
for sent in sincere[:64674*4]:
    a.append(glove_model(sent).vector)
    y.append(0)



XGBOOST_model = GradientBoostingClassifier().fit(a, y)

    
X_test = []
for a in sentences_test:    
    X_test.append(glove_model(a).vector)

preds = XGBOOST_model.predict(X_test)
preds = [int(x) for x in preds]
sub = pd.read_csv("../input/sample_submission.csv")
sub["prediction"] = preds
sub.to_csv("submission.csv",index=False)

#import os
#print(os.listdir("../input"))
