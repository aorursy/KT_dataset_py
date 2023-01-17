import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn import metrics
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.isna().sum()
train.describe()
train.shape
train.info()
X_train = train.drop(['id', 'target'], axis=1)

y_train = train['target']

X_test = test.drop(['id'], axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 500, n_jobs=-1, max_depth=3)
model.fit(X_train, y_train)
Pred = model.predict(X_test)
submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = Pred

submission.to_csv('submission.csv', index=False)