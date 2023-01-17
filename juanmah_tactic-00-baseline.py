import numpy as np

import pandas as pd



import warnings

warnings.simplefilter(action = 'ignore', category = FutureWarning)



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



from lwoku import get_accuracy, get_prediction, plot_confusion_matrix, plot_features_importance
RANDOM_STATE = 1

N_JOBS = -1

N_ESTIMATORS = 2000
X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')

X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')
y_train = X_train['Cover_Type'].copy()
X_train = X_train.drop(['Cover_Type'], axis='columns')
rf_clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,

                                min_samples_leaf=100,

                                verbose=1,

                                random_state=RANDOM_STATE,

                                n_jobs=N_JOBS)
rf_clf.fit(X_train, y_train)
y_test_pred = pd.Series(rf_clf.predict(X_test), index=X_test.index)
y_train_pred = get_prediction(rf_clf, X_train, y_train)
accuracy_score(y_train, y_train_pred)
plot_confusion_matrix(y_train, y_train_pred)
plot_features_importance(X_train.columns, rf_clf)
y_test_pred.to_csv('submission.csv', header=['Cover_Type'], index=True, index_label='Id')