#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 18:31:44 2018

@author: jeffrey
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from collections import Counter
import matplotlib.pyplot as plt
#import seaborn as sns
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import FeatureUnion

train = pd.read_csv('../input/train.tsv',  sep="\t")
test = pd.read_csv('../input/test.tsv',  sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv')

tfhash = [("tfidf", TfidfVectorizer(stop_words='english')),("hashing", HashingVectorizer (stop_words='english'))]
X_train = FeatureUnion(tfhash).fit_transform(train.Phrase)
X_test = FeatureUnion(tfhash).transform(test.Phrase)
y = train.Sentiment
sub['Sentiment'] = LinearSVC(dual=False).fit(X_train,y).predict(X_test) 
sub.to_csv("submission.csv", index=False)
