# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier #loading random forest classifier library
from sklearn import datasets
import json
# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *
np.random.seed(0)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pd_train = pd.read_json('../input/train.json', orient='columns')
pd_test = pd.read_json('../input/test.json', orient='columns')
# for checking data stracture @ pandas
pd.set_option('display.max_columns', 50)
pd.set_option("display.max_rows", 50)
print(pd_train.describe())
pd_train.head(20)
# number of observations for both datasets
print('Observations in training data: ', len(pd_train))
print('Observations in test data: ',len(pd_test))
#setting up x variable as dataset, chosing numerical data only
dataset = pd_train.columns[[1, 2, 5, 9, 10,  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27]]
dataset
y = pd.factorize(pd_train['requester_received_pizza'])[0]
#y = pd_train.iloc[:,22] #requester_received_pizza
y
#creating classifier
classifier = RandomForestClassifier(n_estimators=10, n_jobs=2, random_state = 0)
classifier.fit(pd_train[dataset], y)
classifier.predict_proba(pd_train[dataset])[0:20]