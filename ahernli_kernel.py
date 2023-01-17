import numpy as np

import pandas as pd



from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix



from sklearn.datasets import load_iris
data = load_iris()

iris = pd.DataFrame(data.data, columns=data.feature_names)

iris['target'] = data.target

iris
best_score = 0 

best_params = 0

best_i = 0

best_clf = None



X_train, X_test, y_train, y_test = train_test_split(iris[data.feature_names], iris['target'], test_size=0.2, random_state=42)

param_grid = {'max_depth': np.arange(1, 6), 'random_state': np.arange(0, 50)}

for i in range(2, 11):

    clf = GridSearchCV(DecisionTreeClassifier(presort=True), param_grid, cv=i)

    clf.fit(X_train, y_train)

    print(i, clf.best_params_, clf.best_score_)

    if clf.best_score_ > best_score:

        best_i, best_params, best_score, best_clf = i, clf.best_params_, clf.best_score_, clf

print(best_i, best_params, best_score)
y_pred = best_clf.predict(X_test)



com = confusion_matrix(y_test, y_pred)

com
y_test = np.array(y_test)

y_test
count = len(y_test)

error = 0

for i in range(count):

    if y_pred[i] != y_test[i]:

        error += 1

print('right rate: %s' % ((count-error)/count))