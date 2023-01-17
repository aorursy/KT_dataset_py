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
dataset = pd.read_csv('../input/diabetes.csv')

dataset.head()
from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LogisticRegression

array = dataset.values

X = array[:, 0:8]

y = array[:, 8]



kfold = KFold(n_splits=10, random_state=7)

model = LogisticRegression()

results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print("Accuracy: %.3f (%.df)" % (results.mean(), results.std()))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



model = LinearDiscriminantAnalysis()

results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print("Accuracy: %.3f (%.df)" % (results.mean(), results.std()))
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print("Accuracy: %.3f (%.df)" % (results.mean(), results.std()))
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print("Accuracy: %.3f (%.df)" % (results.mean(), results.std()))
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print("Accuracy: %.3f (%.df)" % (results.mean(), results.std()))
from sklearn.svm import SVC

model = SVC()

results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print("Accuracy: %.3f (%.df)" % (results.mean(), results.std()))
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2





test = SelectKBest(score_func=chi2, k=4)

fit = test.fit(X, y)

# summarize scores

np.set_printoptions(precision=3)



print("scores")

print(fit.scores_)

features = fit.transform(X)

print("features")

print(features[0:5, :])
X = features

model = LogisticRegression()

scoring = 'accuracy'

results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print("Accuracy: %.3f (%.df)" % (results.mean(), results.std()))
from sklearn.feature_selection import RFE

model = LogisticRegression()

array = dataset.values

X = array[:, 0:8]

y = array[:, 8]

rfe = RFE(model, 4)

fit = rfe.fit(X,y)

print("Num Features: %d" % fit.n_features_)

print("Selected Features: %s" % fit.support_)

print("Feature Ranking: %s" % fit.ranking_)
reduced_dataset = dataset.loc[:, fit.support_]

reduced_dataset.head()
X = reduced_dataset.values

kfold = KFold(n_splits=10, random_state=7)

model = LogisticRegression()

results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")

print("Accuracy: %.3f (%.df)" % (results.mean(), results.std()))
from sklearn.decomposition import PCA

array = dataset.values

X = array[:, 0:8]

y = array[:, 8]

pca = PCA(n_components=3)

fit = pca.fit(X)

#summarise components

print("Explained Variance: ", fit.explained_variance_ratio_)

print(fit.components_)
fit
kfold = KFold(n_splits=10, random_state=7)

model = LogisticRegression()

X = pca.transform(X)

results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")

print("Accuracy: %.3f (%.df)" % (results.mean(), results.std()))
from sklearn.ensemble import ExtraTreesClassifier

array = dataset.values

X = array[:, 0:8]

y = array[:, 8]

model = ExtraTreesClassifier()

model.fit(X, y)

print(model.feature_importances_)
dataset.columns