import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/creditcard.csv")

# print(df.describe())

print(df['Class'].value_counts())



y = df['Class']

df = df.drop("Class",axis=1)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=0)
from sklearn.ensemble import  AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.naive_bayes import GaussianNB

boost_clf = AdaBoostClassifier() 

param_grid = {'n_estimators': np.arange(40, 70, 10), 'base_estimator': [DecisionTreeClassifier(), 

                                                                        GaussianNB()]}

gs_boost_clf = GridSearchCV(boost_clf, param_grid, n_jobs=-1, verbose=1)
gs_boost_clf = gs_boost_clf.fit(X_train, y_train)

boost_pred = gs_boost_clf.predict(X_test)

print(classification_report(y_test, y_pred=boost_pred))
print('Best score of AdaBoost classifier: %.2f' %gs_boost_clf.best_score_)

for param_name in param_grid.keys():

    print('%s %r' %(param_name, gs_boost_clf.best_params_[param_name]))