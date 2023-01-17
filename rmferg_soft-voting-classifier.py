import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score, accuracy_score





df = pd.read_csv('../input/diabetes.csv')





X = df.loc[:, ['Pregnancies','Glucose','BloodPressure','SkinThickness',

           'Insulin','BMI', 'DiabetesPedigreeFunction','Age']].values

y = df.loc[:, 'Outcome'].values





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
import time

t0 = time.clock()



tree = DecisionTreeClassifier(random_state=1)

svm = SVC(probability=True, kernel='rbf')

knn = KNeighborsClassifier(p=2, metric='minkowski')

nb = GaussianNB()

eclf = VotingClassifier(estimators=[('tree', tree), ('svm', svm), ('knn', knn),('nb', nb)], voting='soft')

param_range10 = [.001, .01, 1, 10, 100]

param_range1 = list(range(3, 8))

param_grid = [{'svm__C':param_range10, 'svm__gamma':param_range10, 'tree__max_depth':param_range1, 

               'knn__n_neighbors':param_range1}]



gs = GridSearchCV(estimator=eclf, param_grid=param_grid, scoring='accuracy', cv=5)

gs = gs.fit(X_train_std, y_train)



print('Best accuracy score: %.3f \nBest parameters: %s' % (gs.best_score_, gs.best_params_))



clf = gs.best_estimator_

clf.fit(X_train_std, y_train)

t1 = time.clock()

print('Running time: %.3f' % (t1-t0))
from sklearn.metrics import confusion_matrix



y_pred = clf.predict(X_test_std)

print('ROC AUC: %.3f \nAccuracy: %.3f \nConfusion Matrix:' % (roc_auc_score(y_true=y_test, y_score=y_pred),

                                         accuracy_score(y_true=y_test, y_pred=y_pred)))

print(confusion_matrix(y_true=y_test, y_pred=y_pred))