import numpy as np
import pandas as pd
df = pd.read_csv('../input/train.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
from sklearn.cross_validation import train_test_split
X= df.iloc[:,:-1]
X
Y = df.iloc[:,-1:]
Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
lr.fit(X_train, Y_train)
Y_pred= lr.predict(X_test)
lr.score(X_test, Y_test)
test = pd.read_csv('../input/test.csv')
test.head()
test_pred = lr.predict(test)
pd.DataFrame(test_pred)
submission = pd.read_csv('../input/sample_submission.csv')
submission.head()
submission['winPlacePerc']=test_pred
submission.head()
submission.to_csv('submission.csv', index=False)
