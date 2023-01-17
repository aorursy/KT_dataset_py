import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn import preprocessing
train = pd.read_csv('/kaggle/input/kpitmovies/train_data.csv')
train.head(2)
basic_features = ['runtime', 'budget', 'revenue', 'vote_count']
basic_X = train[basic_features]
basic_y = train['target']
assert basic_X.shape[0] == basic_y.shape[0]
basic_X_train, basic_X_validate, basic_y_train, basic_y_validate = train_test_split(basic_X, basic_y)
logres =  LogisticRegression()
logres.fit(basic_X_train, basic_y_train)
logres_y_pred = logres.predict(basic_X_validate)
print('Accuracy / train:\t',cross_val_score(logres, basic_X_train, basic_y_train).mean())
print('Accuracy / validation:  ',accuracy_score(logres_y_pred, basic_y_validate))
#grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
#logreg=LogisticRegression()
#logreg_cv=GridSearchCV(logreg,grid,cv=10)
#logreg_cv.fit(basic_X_train,basic_y_train)
#print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
#print("accuracy :",logreg_cv.best_score_)
logreg2=LogisticRegression(C=0.001,penalty="l2")
logreg2.fit(basic_X_train,basic_y_train)
print("score",logreg2.score(basic_X_train,basic_y_train))
tree =  DecisionTreeClassifier()
tree.fit(basic_X_train, basic_y_train)
tree_y_pred = tree.predict(basic_X_validate)
print('Accuracy / train:\t', cross_val_score(tree, basic_X_train, basic_y_train).mean())
print('Accuracy / validation:  ', accuracy_score(basic_y_validate,tree_y_pred))
#grid1={"max_depth":np.logspace(-3, 3, 7), "min_samples_split":[2], "max_features":np.ndarray([4])}
#tree1 = DecisionTreeClassifier()
#tree_cv=GridSearchCV(tree,grid1,cv=10)
#tree_cv.fit(basic_X_train,basic_y_train)
#print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
#print("accuracy :",tree_cv.best_score_)
tree2=DecisionTreeClassifier(max_depth=10, max_features=1.0, min_samples_split=2)
tree2.fit(basic_X_train,basic_y_train)
print("accuracy",tree2.score(basic_X_train,basic_y_train))
knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))])
knn_params = {'knn__n_neighbors': range(1, 30)}
knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True)
knn_grid.fit(basic_X_train, basic_y_train)
print("accuracy", knn_grid.score(basic_X_train, basic_y_train))
#grid1={"max_depth":np.logspace(-3, 3, 7), "min_samples_split":[2], "max_features":np.ndarray([4])}
#tree1 = DecisionTreeClassifier()
#tree_cv=GridSearchCV(tree,grid1,cv=10)
#tree_cv.fit(basic_X_train,basic_y_train)
#print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
#print("accuracy :",tree_cv.best_score_)
grid1={"n_estimators":[50], "max_depth":np.logspace(-3, 3, 7), "min_samples_split":[2], "max_features":np.ndarray([4])}
forest = RandomForestClassifier()
forest_cv=GridSearchCV(forest,grid1,cv=10)
forest_cv.fit(basic_X_train,basic_y_train)
print("tuned hpyerparameters :(best parameters) ",forest_cv.best_params_)
print("accuracy :",forest_cv.best_score_)
grid_forest=RandomForestClassifier(max_depth=10, max_features=1.0, min_samples_split=2, n_estimators=50)
grid_forest.fit(basic_X_train,basic_y_train)
print("accuracy",grid_forest.score(basic_X_train,basic_y_train))
test = pd.read_csv('/kaggle/input/kpitmovies/test_data.csv').drop(61)
test.head(2)
test.movie_id = test.movie_id.astype('int')
basic_X_test = test[basic_features]
basic_X_test.head(2)
#tree_prediction = tree.predict(basic_X_test)
#grid_tree_prediction = tree2.predict(basic_X_test)
#knn_grid_prediction = knn_grid.predict(basic_X_test)
#forest_prediction = forest.predict(basic_X_test)
grid_forest_prediction = grid_forest.predict(basic_X_test)
#grid_tree_prediction
#forest_prediction
#grid_forest_prediction
pomogite = pd.read_csv('/kaggle/input/kpitmovies/sample_submission.csv')
pomogite.movie_id = test.movie_id.values
pomogite.target = grid_forest_prediction
#pomogite.to_csv('1.csv', index=False)