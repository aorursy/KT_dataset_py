import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

np.set_printoptions(edgeitems=10)

print(os.listdir("../input"))
train = pd.read_json('../input/train.json')
print(train.shape)
train.head()
test = pd.read_json('../input/test.json')
print(test.shape)
test.head()
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()
from sklearn.feature_extraction.text import CountVectorizer

X = [' '.join([ingredient.replace(' ', '_') for ingredient in x.ingredients]) for _, x in train.iterrows()]
X_test = [' '.join([ingredient.replace(' ', '_') for ingredient in x.ingredients]) for _, x in test.iterrows()]

vectorizer = CountVectorizer()
vectorizer = vectorizer.fit(X + X_test)

X = vectorizer.transform(X)
X_test = vectorizer.transform(X_test)
print(f'Training inputs size: {X.shape}')
print(f'Test inputs size: {X_test.shape}')
from sklearn.preprocessing import OneHotEncoder

y = train.cuisine.values

# y_train = train.cuisine.values.reshape(-1, 1)

# encoder = OneHotEncoder()
# encoder = encoder.fit(np.concatenate((y_train, y_test)))

# y_train = encoder.transform(y_train)

print(f'Training target size: {y.shape}')
split = int(0.8 * X.shape[0])  # Use 80% of the data for training and 20% for validation.
print(f'Training, validation split index: {split}')
# Perform training, validation datasets split.
X_train, y_train, X_valid, y_valid = X[:split], y[:split], X[split:], y[split:]
print(f'Training inputs size: {X_train.shape}')
print(f'Training target size: {y_train.shape}')
print(f'Validation inputs size: {X_valid.shape}')
print(f'Validation target size: {y_valid.shape}')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(X_train, y_train)
train_preds = model.predict(X_train)
valid_preds = model.predict(X_valid)

print(f'Training classification accuracy: {accuracy_score(y_train, train_preds)}')
print(f'Validation classification accuracy: {accuracy_score(y_valid, valid_preds)}')
model.fit(X, y)
train_preds = model.predict(X)

print(f'Training classification accuracy: {accuracy_score(y, train_preds)}')
test_preds = model.predict(X_test)
submission = pd.DataFrame()
submission['id'] = test.id
submission['cuisine'] = pd.Series(test_preds)
submission.head()
submission.to_csv('predictions.csv', index=False)