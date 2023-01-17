# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from tpot import TPOTClassifier

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
X_train = train.iloc[:, 2:]

y_train = train.iloc[:, 1]
# 0.768

pipeline_optimizer = TPOTClassifier(generations=10, population_size=200, cv=5, n_jobs=-1,

                                    random_state=42, verbosity=2, early_stop=5)
pipeline_optimizer.fit(X_train, y_train)
test = pd.read_csv('../input/test.csv')
predict = pipeline_optimizer.predict(test.iloc[:, 1:])
sub = pd.read_csv("../input/sample_submission.csv")

sub.target = predict

sub.to_csv('sub.csv', index=False)
sub.head()