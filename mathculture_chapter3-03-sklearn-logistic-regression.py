# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_iris

iris = load_iris()
iris.keys()
print("data", iris.data[0])

print('target', iris.target[0])

print('target_names', iris.target_names)

# 長いので省略

# print('DESCR', iris.DESCR)

print('feature_names', iris.feature_names)

print('filename', iris.filename)
iris.data.shape
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial')

model.fit(X, y)

print("model" , model.predict(X[:2, :]))

print("prob", model.predict_proba(X[:2, :]))
model.score(X, y) # accuracy
from sklearn.model_selection import cross_val_score
model = LogisticRegression(solver='lbfgs', multi_class='multinomial')

scores = cross_val_score(model, X, y)

print('Cross-Validation scores: {}'.format(scores))

print('Average score: {}'.format(np.mean(scores)))