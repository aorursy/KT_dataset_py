import os
import scipy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from time import time

np.set_printoptions(edgeitems=20, linewidth=200)

print(os.listdir("../input"))
train = pd.read_json('../input/train.json')
print(train.shape)
train.head()
test = pd.read_json('../input/test.json')
print(test.shape)
test.head()
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()
from sklearn.feature_extraction.text import TfidfVectorizer

X = [' '.join([ingredient.replace(' ', '_') for ingredient in x.ingredients]) for _, x in train.iterrows()]
X_test = [' '.join([ingredient.replace(' ', '_') for ingredient in x.ingredients]) for _, x in test.iterrows()]

vectorizer = TfidfVectorizer()
vectorizer = vectorizer.fit(X + X_test)

X = vectorizer.transform(X)
X_test = vectorizer.transform(X_test)
print(f'Training inputs size: {X.shape}')
print(f'Test inputs size: {X_test.shape}')
from sklearn.preprocessing import OneHotEncoder

y = train.cuisine.values
print(f'Training target size: {y.shape}')
split = int(0.8 * X.shape[0])  # Use 80% of the data for training and 20% for validation.
print(f'Training, validation split index: {split}')
# Perform training, validation datasets split.
X_train, y_train, X_valid, y_valid = X[:split], y[:split], X[split:], y[split:]
print(f'Training inputs size: {X_train.shape}')
print(f'Training target size: {y_train.shape}')
print(f'Validation inputs size: {X_valid.shape}')
print(f'Validation target size: {y_valid.shape}')
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression(max_iter=10)
param_dist = {'C' : scipy.stats.expon(scale=1.0)}
n_iter = 25
search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=n_iter)

start = time()
search.fit(X, y)
print(f'RandomizedSearchCV took {time() - start:.2f} seconds for {n_iter} candidates parameter settings.')
print(search.cv_results_)
model = search.best_estimator_
train_preds = model.predict(X)

print(f'Training classification accuracy: {accuracy_score(y, train_preds)}')
test_preds = model.predict(X_test)
submission = pd.DataFrame()
submission['id'] = test.id
submission['cuisine'] = pd.Series(test_preds)
submission.head()
submission.to_csv('predictions.csv', index=False)