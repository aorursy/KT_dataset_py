import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
train.head()
X = train.drop('winPlacePerc', 1)
y = train['winPlacePerc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True, n_jobs=8)

lreg = model.fit(X_train, y_train)
print("Train Score:", lreg.score(X_train, y_train))
print("Test Score:", lreg.score(X_test, y_test))
test = pd.read_csv('../input/test.csv')

test.head()
pred = lreg.predict(test)
submission = pd.DataFrame.from_dict(data={'Id': test['Id'], 'winPlacePerc': pred})

submission.head()
submission.to_csv('submission.csv', index=False)
