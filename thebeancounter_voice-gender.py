# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

l = LabelEncoder()
s = SVC(kernel="sigmoid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

data = pd.read_csv("../input/voice.csv")
x = data.drop("label", axis=1)
y = l.fit_transform(data["label"])


# Any results you write to the current directory are saved as output.
data["label"].value_counts()
#dataset is balanced 

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf', "linear", "sigmoid"]},
 ]

from sklearn.model_selection import GridSearchCV
g = GridSearchCV(s, param_grid[1], n_jobs=-1, cv=5)
g.fit(x, y)

print(g.cv_results_.keys())
print(max(g.cv_results_["mean_test_score"]))
print(g.best_estimator_)