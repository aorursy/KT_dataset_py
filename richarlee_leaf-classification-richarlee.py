import time

import pandas as pd

import numpy as np

from pandas import DataFrame,Series



from sklearn import linear_model, cross_validation, feature_selection, manifold, decomposition, random_projection

from sklearn.preprocessing import MinMaxScaler,LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier,BaggingRegressor,RandomForestClassifier

from sklearn.learning_curve import learning_curve

from sklearn.cross_validation import StratifiedKFold

from sklearn.svm import LinearSVC,SVC

from sklearn.metrics import log_loss

#import所需的package

from sklearn.multiclass import OneVsRestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.grid_search import GridSearchCV



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

train_df.fillna(0,inplace=True)

train_df
le = LabelEncoder().fit(train_df.species)

labels = le.transform(train_df.species)

labels
df = train_df.copy()

df.species = labels

df.species
df.ix[:,2:] = MinMaxScaler().fit_transform(train_df.ix[:,2:])

df
X = df.as_matrix()[:,2:]

y = df.as_matrix()[:,1]
params = {'C':[1500, 2000, 2500], 'tol': [0.0001]}

# solver='newton-cg' or 'lbfgs'

log_reg = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=400)

clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='True', n_jobs=-1, cv=5)

clf.fit(X, y)



print("best params: " + str(clf.best_params_))

for params, mean_score, scores in clf.grid_scores_:

  print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std(), params))

  print(scores)
estimator = clf

# estimator.fit(X,y)
test_data = pd.read_csv('../input/test.csv')

test_df = DataFrame(MinMaxScaler().fit_transform(test_data.ix[:,1:]))

test_df
species = train_df.species.unique()

species.sort()



predict = estimator.predict_proba(test_df.as_matrix())

result = DataFrame(predict,columns=species)

result

# train_df.species

# print(predict)

# decision = estimator.decision_function(test_df.as_matrix())

# decision.shape
result.insert(0,'id',test_data.id)

result.to_csv('result.csv',index=False)