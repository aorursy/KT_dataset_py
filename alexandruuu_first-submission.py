# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import string
import operator
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv("../input/train.csv",header=None,low_memory=False)
data_test = pd.read_csv("../input/test.csv",header=None,low_memory=False)

Y = data[2][1:]
Y = Y.values
print(Y.shape)

sentences = data[1][1:]
sentences_test = data_test[1][1:]

vectorizer = CountVectorizer(min_df=200)
X = vectorizer.fit_transform(list(sentences))
X_test = vectorizer.transform(list(sentences_test))

LR_model = LogisticRegression(random_state=0, solver='lbfgs').fit(X, Y)
preds = LR_model.predict(X_test)


import os
print(os.listdir("../input"))

preds = [int(x) for x in preds]
sub = pd.read_csv("../input/sample_submission.csv")
sub["prediction"] = preds
sub.to_csv("submission.csv", index=False)


# Any results you write to the current directory are saved as output.
