import pandas as pd

from sklearn.linear_model import LogisticRegression





train = pd.read_csv("../input/train.csv").drop('id', axis=1)



y_train = train['target']

X_train = train.drop('target', axis=1)



test = pd.read_csv('../input/test.csv')

X_test = test.drop('id', axis = 1)



submission = pd.read_csv('../input/sample_submission.csv')





clf = LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='liblinear').fit(X_train, y_train)



submission['target'] = clf.predict_proba(X_test)[:,1]

submission.to_csv('submission.csv', index=False)