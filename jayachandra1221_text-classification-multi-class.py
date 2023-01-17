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
import nltk
from __future__ import division
import math
import string
import pandas as pd
import numpy as np
train=pd.read_csv("../input/train.tsv",delimiter='\t')
test=pd.read_csv("../input/test.tsv",delimiter='\t')
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
train.groupby('Sentiment').Phrase.count().plot.bar(ylim=0)
plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(train.Phrase).toarray()
labels = train.Sentiment
features.shape
from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(features, labels)
features_test = tfidf.transform(test.Phrase)

features_test.shape
pred = model.predict(features_test)
submission=pd.read_csv("../input/test.tsv",delimiter='\t')
submission['sentiment']=pred
submission.drop(['SentenceId', 'Phrase'], axis=1, inplace=True)
submission
submission.to_csv("submission.csv", index = False)
