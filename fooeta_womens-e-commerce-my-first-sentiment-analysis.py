import os
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
base = pd.read_csv(os.path.join("../input", "Womens Clothing E-Commerce Reviews.csv"))
base.info()
base = base.dropna(subset=["Review Text"])
from sklearn.model_selection import train_test_split
X = base.drop('Recommended IND', axis=1)
y = base['Recommended IND'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(min_df=5)
vect.fit(X_train['Review Text'])
X_train_review = vect.transform(X_train['Review Text'])
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

scores = cross_val_score(LogisticRegression(), X_train_review, y_train, cv=5)
print("Mean cross-val accuracy: {:.2f}".format(np.mean(scores)))
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train_review, y_train)
print("Best cross-val score: {:.2f}".format(grid.best_score_))
print("Best params: ", grid.best_params_)
X_train['review_pred'] = grid.predict(X_train_review)

X_test['review_pred'] = grid.predict(vect.transform(X_test['Review Text']))
X_train_dummies = pd.get_dummies(X_train.drop(['Unnamed: 0', 'Clothing ID', 'Review Text'], axis=1))
X_train_dummies.info()
from sklearn.ensemble import RandomForestClassifier

param_grid2 = {'n_estimators': [100, 200, 300],
               'max_depth': [1, 2, 3]}
grid2 = GridSearchCV(RandomForestClassifier(random_state=0), param_grid2, cv=5)
grid2.fit(X_train_dummies, y_train)
print("Best cross-val score: {:.2f}".format(grid2.best_score_))
print("Best params: ", grid2.best_params_)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_train_dummies, y_train)
X_pca = pca.transform(X_train_dummies)

colors = []
for i in y_train:
    if i == 0:
        colors += ['red']
    else:
        colors += ['cyan']
plt.scatter(X_pca[:,0], X_pca[:,1], color=colors, alpha=.2)
