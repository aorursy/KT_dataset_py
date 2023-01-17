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
from sklearn.model_selection import GroupKFold

gkf = list(GroupKFold(n_splits=5).split(X, y, train.check_id.values))
score = cross_val_score(LogisticRegression(), X, y, scoring='neg_log_loss', cv = gkf, n_jobs = -1)
"%.3f +- %.4f" % (-np.mean(score), np.std(score))
from sklearn.model_selection import cross_val_predict

predicts = cross_val_predict(LogisticRegression(), X, y, cv = gkf, n_jobs = -1, method = 'predict_proba')
predicts[np.arange(len(predicts)), y] = 1 - predicts[np.arange(len(predicts)), y]
import matplotlib.pyplot as plt

plt.figure(figsize = (15, 8))
_ = plt.hist(np.log(1.0 - predicts.ravel()), bins = 50, log = True)
train['category_mistake'] = predicts[np.arange(len(predicts)), y]
train['p_category'] = cross_val_predict(LogisticRegression(), X, y, cv = gkf, n_jobs = -1, method = 'predict')
train['p_category'] = labeler.inverse_transform(train['p_category'])
train.sort_values('category_mistake', ascending = False)[['name', 'category', 'category_mistake', 'p_category']]
