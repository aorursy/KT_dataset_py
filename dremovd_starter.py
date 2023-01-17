import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv.gz')
train_checks = pd.read_csv('../input/train_checks.csv.gz')
print(train.shape, train_checks.shape)
print(train.columns.values)
print(train_checks.columns.values)
train.fillna('', inplace=True)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train.name)
for i, word in enumerate(vectorizer.vocabulary_):
    print(word)
    if i > 10:
        break
X.shape
from sklearn.preprocessing import LabelEncoder

labeler = LabelEncoder()
y = labeler.fit_transform(train.category)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupShuffleSplit

gkf = list(GroupShuffleSplit(n_splits=5, random_state=0).split(X, y, train.check_id.values))
score = cross_val_score(LogisticRegression(), X, y, scoring='neg_log_loss', cv = gkf, n_jobs = -1)
"%.3f +- %.4f" % (-np.mean(score), np.std(score))
model = LogisticRegression()
model.fit(X, y)
test = pd.read_csv('../input/test.csv.gz')
test_checks = pd.read_csv('../input/test_checks.csv.gz')
X_test = vectorizer.transform(test.name)
p_test = model.predict_proba(X_test)
test = test[['id']]
for i, c in enumerate(labeler.classes_):
    test[c] = p_test[:, i]
test.to_csv('submission.csv.gz', compression='gzip', encoding = 'utf-8', index = False)
