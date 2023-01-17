import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv.gz')
test = pd.read_csv('../input/test.csv.gz')
train_checks = pd.read_csv('../input/train_checks.csv.gz')
test_checks = pd.read_csv('../input/test_checks.csv.gz')
train = train.merge(train_checks, on = 'check_id', how = 'left')
test = test.merge(test_checks, on = 'check_id', how = 'left')
print(train.shape, train_checks.shape)
print(train.columns.values)
print(train_checks.columns.values)
print(test.shape, test_checks.shape)
print(test.columns.values)
train.fillna('', inplace=True)
test.fillna('', inplace=True)
catalog = pd.read_csv('../input/catalog2.csv.gz')
catalog.shape
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(train['name'])
X_test = vectorizer.transform(test['name'])
X_catalog = vectorizer.transform(catalog.description.fillna(""))
from sklearn.preprocessing import LabelEncoder

catalog_labeler = LabelEncoder()
y_catalog = catalog_labeler.fit_transform(catalog.category)
X.shape
from sklearn.preprocessing import LabelEncoder

labeler = LabelEncoder()
y = labeler.fit_transform(train.category)
from sklearn.metrics import log_loss, make_scorer
clipping = 0.001

clipped_log_loss = make_scorer(log_loss, eps = clipping, greater_is_better = False, needs_proba = True)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score

parameters = {
    'C' : np.logspace(0, 3, 4),
}

gkf = list(GroupKFold(n_splits=4).split(X, y, train.check_id.values))
score = cross_val_score(LogisticRegression(C = 100), X, y, cv = gkf, scoring=clipped_log_loss)
-np.mean(score)
from sklearn.model_selection import cross_val_predict
X_meta = cross_val_predict(LogisticRegression(C = 100), X, y, cv=gkf, n_jobs = -1, method = 'predict_proba')
X_meta.shape
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators = 40)

score = cross_val_score(xgb, X_meta, y, cv = gkf, scoring=clipped_log_loss)
-np.mean(score)
model_catalog = LogisticRegression(C = 100)
model_catalog.fit(X_catalog, y_catalog)

X_meta_catalog = model_catalog.predict_proba(X)
X_meta_catalog.shape
score = cross_val_score(xgb, np.hstack([X_meta, X_meta_catalog]), y, cv = gkf, scoring=clipped_log_loss)

-np.mean(score)
X_meta = np.zeros((X.shape[0], 25))
X_test_meta = []

for fold_i, (train_i, test_i) in enumerate(gkf):
    print(fold_i)
    model = LogisticRegression(C = 100)
    model.fit(X.tocsr()[train_i], y[train_i])
    X_meta[test_i, :] = model.predict_proba(X.tocsr()[test_i])
    X_test_meta.append(model.predict_proba(X_test))
X_test_meta = np.stack(X_test_meta)
X_test_meta.shape
X_test_meta_mean = np.mean(X_test_meta, axis = 0)
X_test_meta_mean.shape
X_meta = np.hstack([X_meta, X_meta_catalog])
X_test_meta_catalog = model_catalog.predict_proba(vectorizer.transform(test.name))
X_test_meta = np.hstack([X_test_meta_mean, X_test_meta_catalog])
xgb.fit(X_meta, y)
p_test = xgb.predict_proba(X_test_meta)
def form_predictions(p):
    return ['%.6f' % x for x in p]
test_submission = test[['id']]

for i, c in enumerate(labeler.classes_):
    p = p_test[:, i]
    p[p < clipping] = clipping
    p[p > (1.0 - clipping)] = (1.0 - clipping)
    test_submission[c] = form_predictions(p)
test_submission.to_csv('meta_model_extended.csv.gz', compression='gzip', index = False)