# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
id_col = training_data.columns[0]
feature_cols = list(training_data.columns[1:-1])
target_col = training_data.columns[-1]
from sklearn.decomposition import PCA
pca = PCA(n_components=14).fit(training_data[feature_cols]) # 14 PCs explain virtually all the variance

# Print the components and the amount of variance in the data contained in each dimension
print ('\n', pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
X_train = pca.transform(balanced_data[feature_cols])
y_train = balanced_data[target_col]

X_test = pca.transform(test_data[feature_cols])
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
clf = SVC()
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV

def best(classifier, paramdict, X_train, y_train):
    kfcv = KFold(n=len(y_train), n_folds=10, shuffle=True)
    gs = GridSearchCV(classifier, paramdict, cv=kfcv)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_

params = {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
clf = SVC()
best_model, best_hyperparams = best(clf, params, X_train, y_train)

print(best_hyperparams)
clf = best_model
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
results = pd.DataFrame({'TARGET':predictions}, index=test_data[id_col])
print(results[results['TARGET']==1].shape)
print(results)
results.to_csv('submission.csv')
