import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np

# 先了解自己的数据
train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])
test = pd.read_csv('../input/test.csv', parse_dates=['Dates'])

train.head()
test.head()
all_addr = np.array(train.Address.tolist() + test.Address.tolist())

list(all_addr)
stop_words = ['dr', 'wy', 'bl', 'av', 'st', 'ct', 'ln', 'block', 'of']
vectorizer = CountVectorizer(max_features=300, stop_words=stop_words)
features = vectorizer.fit_transform(all_addr).toarray()
features[0,:]
X = features[:train.shape[0]]
y = train.Category
X
y
#分成80%的训练集和20%的验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

log_model = LogisticRegression().fit(X=X_train, y=y_train)

results = log_model.predict_proba(X_test)

np.round(results[1], 3)
log_loss_score = log_loss(y_test, results)
print('log loss score: {0}'.format(round(log_loss_score, 3)))
log_model = LogisticRegression().fit(X=features[:train.shape[0]], y=train.Category)
results = log_model.predict_proba(features[train.shape[0]:])
results
submission = pd.DataFrame(results)
submission.columns = sorted(train.Category.unique())
submission.set_index(test.Id)
submission.index.name="Id"
submission.to_csv('../working/py_submission_logreg_addr_300.csv')
