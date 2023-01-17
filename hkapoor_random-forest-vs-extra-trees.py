import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data_X = pd.DataFrame(data["data"], columns=data['feature_names'])
data_X
data_Y = data["target"]
data_Y[:30]
data_X.shape
data_Y.shape
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
etc = ExtraTreesClassifier(random_state=0)
cv_score = cross_val_score(etc, data_X, data_Y, cv=5).mean()
print("The avergae Cross validation score is ", cv_score)
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[50, 100, 500, 1000], 'max_depth':[5, 10, 50, 100, 500]}
etc = ExtraTreesClassifier()
clf = GridSearchCV(etc, parameters, cv=5)
clf.fit(data_X, data_Y)
clf.best_estimator_
clf.best_score_
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import time
start_time = time.time()

dtc = DecisionTreeClassifier(random_state=0)
cv_score = cross_val_score(dtc, data_X, data_Y, cv=5).mean()
print("Average Cross Validation Score = ", cv_score)
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

rfc = RandomForestClassifier(random_state=0)
cv_score = cross_val_score(rfc, data_X, data_Y, cv=5).mean()
print("Average Cross Validation Score = ", cv_score)
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

etc = ExtraTreesClassifier(random_state=0)
cv_score = cross_val_score(etc, data_X, data_Y, cv=5).mean()
print("Average Cross Validation Score = ", cv_score)
print("--- %s seconds ---" % (time.time() - start_time))