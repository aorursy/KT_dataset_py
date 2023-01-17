import pandas as pd

from sklearn.naive_bayes import GaussianNB

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

features = [x for x in train.columns if 'var_' in x]

clf = GaussianNB()

clf.fit(train[features], train['target'])

test['target'] = clf.predict_proba(test[features])[:, 1]

test[['ID_code', 'target']].to_csv('GaussianNB_submission.csv', index=False)