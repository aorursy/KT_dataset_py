import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
from sklearn.neighbors import KNeighborsClassifier
train = pd.read_csv('/kaggle/input/kpitmovies/train_data.csv')

train.head(2)
basic_features = ['runtime', 'budget', 'revenue', 'vote_count']
basic_X = train[basic_features]
basic_y = train['target']
basic_X.head()
basic_y.head()
assert basic_X.shape[0] == basic_y.shape[0]
basic_X_train, basic_X_validate, basic_y_train, basic_y_validate = train_test_split(basic_X, basic_y)
logres =  LogisticRegression()
logres.fit(basic_X_train, basic_y_train)
logres_y_pred = logres.predict(basic_X_validate)
print('Accuracy / train:\t',cross_val_score(logres, basic_X_train, basic_y_train).mean())
print('Accuracy / validation:  ',accuracy_score(logres_y_pred, basic_y_validate))
tree =  DecisionTreeClassifier()
tree.fit(basic_X_train, basic_y_train)
tree_y_pred = tree.predict(basic_X_validate)
print('Accuracy / train:\t', cross_val_score(tree, basic_X_train, basic_y_train).mean())
print('Accuracy / validation:  ', accuracy_score(basic_y_validate,tree_y_pred))
knn = KNeighborsClassifier()
knn.fit(basic_X_train, basic_y_train)
knn_y_pred = knn.predict(basic_X_validate)
print('Accuracy / train:\t', cross_val_score(knn, basic_X_train, basic_y_train).mean())
print('Accuracy / validation:  ', accuracy_score(basic_y_validate,knn_y_pred))
poll = VotingClassifier(estimators=[('logres', logres),('dt', tree),('knn', knn)], weights=[1, 1, 1], voting='hard')
poll.fit(basic_X_train, basic_y_train)
poll_y_pred = poll.predict(basic_X_validate)
print('Accuracy / train:\t', cross_val_score(poll, basic_X_train, basic_y_train).mean())
print('Accuracy / validation:  ', accuracy_score(basic_y_validate,poll_y_pred))
test = pd.read_csv('/kaggle/input/kpitmovies/test_data.csv').drop(61)
test.head(2)
test.movie_id = test.movie_id.astype('int')
basic_X_test = test[basic_features]
basic_X_test.head(2)
tree_prediction = tree.predict(basic_X_test)
poll_prediction = poll.predict(basic_X_test)
#tree_prediction
submission = pd.read_csv('/kaggle/input/kpitmovies/sample_submission.csv')
submission.head()
#test.movie_id.values
submission.movie_id = test.movie_id.values
submission.target = tree_prediction
#submission.to_csv('tree_baseline.csv', index=False)
submission.target = poll_prediction
#submission.to_csv('poll_baseline.csv', index=False)