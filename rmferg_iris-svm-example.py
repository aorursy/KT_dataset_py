import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.svm import SVC

from sklearn.cross_validation import train_test_split, cross_val_score



df = pd.read_csv('../input/Iris.csv')

X = df.loc[:, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

y = df.loc[:, 'Species'].values



le = LabelEncoder()

y = le.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)
clf = SVC(kernel='linear', decision_function_shape = 'ovr')

clf.fit(X_train_std, y_train)

scores = cross_val_score(estimator=clf, X=X_train_std, y=y_train, cv=5, n_jobs=1)

print('CV accuracy scores: %s' % scores)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
from sklearn.grid_search import GridSearchCV

param_range = [10.0**x for x in range(-4, 4)]

param_grid = [{'C' : param_range,

               'kernel': ['linear']},

                {'C': param_range,

                 'gamma': param_range,

                 'kernel': ['rbf']}]

gs = GridSearchCV(SVC(decision_function_shape = 'ovr'),

                 param_grid=param_grid,

                 scoring='accuracy',

                 cv=5,

                 n_jobs=1)

gs = gs.fit(X_train_std, y_train)

clf1 = gs.best_estimator_

clf1.fit(X_train_std, y_train)

print('Best score: %.3f; Best params: %s \nTest accuracy: %.3f' % (gs.best_score_, gs.best_params_,

                                                                   clf1.score(X_test_std, y_test)))