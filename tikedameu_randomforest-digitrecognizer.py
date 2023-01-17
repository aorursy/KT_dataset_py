"""

    This kernel is inspired by https://www.kaggle.com/hideki1234/randomforest-of-tree-and-accuracy

"""

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import gc

import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
X_tr = df_train.iloc[:,1:]

y_tr = df_train.iloc[:, 0]



print(X_tr.head())
means = []

stds = []

n_trees = [10, 15, 20, 25, 30, 40, 50, 70, 100]

for t in n_trees:

    print(t)

    recognizer = RandomForestClassifier(t)

    score = cross_val_score(recognizer, X_tr, y_tr)

    print(score)

    means.append(np.mean(score))

    stds.append(np.std(score))
means = np.array(means)

stds = np.array(stds)

print(means)

print(stds)
plt.plot(n_trees, means)

plt.plot(n_trees, means+stds, 'b--')

plt.plot(n_trees, means-stds, 'b--')

plt.ylabel('Cross Validation score')

plt.xlabel('# of trees')

plt.savefig('cv_trees.png')
X_ts = df_test
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

from sklearn.datasets import make_classification



# X_tr, y_tr = make_classification(n_samples=len(X_tr), n_features=784,

#    n_informative=2, n_redundant=0,

#    random_state=0, shuffle=False)



clf = RandomForestClassifier(n_estimators=100,

                             random_state=0)



clf.fit(X_tr, y_tr)
y_ts = clf.predict(X_ts)



my_submission = pd.DataFrame({'ImageId': [i for i in range(1, len(y_ts)+1)], 'Label': y_ts})



print(my_submission.head())
my_submission.to_csv('submission.csv', index=False)