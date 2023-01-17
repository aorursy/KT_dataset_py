# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load Training Data

df_train = pd.read_csv('../input/train.csv', index_col = 'id')

df_train.head()
# Split Training data into X and y

X = df_train.drop('species', axis=1)

y = df_train.species

df_train.shape, X.shape, y.shape
# Examine Classes

classes = df_train.species.unique()

classes.sort()

classes[:5]
# Binary Encode Class Labels into y

import sklearn.preprocessing as skpp

y = skpp.label_binarize(y, classes = classes)

y[5,]
# Scale all parameters in X

import sklearn.preprocessing as skpp



#scaler = skpp.StandardScaler()

#X = scaler.fit_transform(X)


# Create Data Shuffler

import sklearn.model_selection as skms

strat_cv_shuffler = skms.StratifiedShuffleSplit(n_splits=2, train_size=0.8)
# Create Classifier

from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier

svm = OneVsRestClassifier(SVC(kernel='linear', probability=True))
# Run on Training Data and Review Accuracy Scores

scores = skms.cross_val_score(svm, X, y, cv=strat_cv_shuffler)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))