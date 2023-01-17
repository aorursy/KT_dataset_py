import pandas as pd
import numpy as np
import sklearn
pd.options.display.float_format = lambda x: f' {x:,.2f}'
import warnings
warnings.filterwarnings("ignore")
titanic = pd.read_csv('../input/train.csv', index_col='PassengerId')
titanic.head()
titanic.corr()
X = titanic['Fare'].values
X[:5]
y = titanic['Survived'].values
y[:5]
X.ndim, y.ndim
X = X.reshape(-1, 1)
X.ndim, y.ndim
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X, y)
titanic_test_data = pd.read_csv('../input/test.csv', index_col='PassengerId')
titanic_test_data.head()
titanic_test_data['Fare'].head()
titanic_test_data['Fare'].isna().sum()
titanic_test_data[['Fare']] = titanic_test_data[['Fare']].fillna(titanic_test_data['Fare'].mean())
titanic_test_data['Fare'].isna().sum()
X_test = titanic_test_data['Fare'].values
X_test[:5]
X_test = X_test.reshape(-1, 1)
X_test[:5]
predictions = logr.predict(X_test).reshape(-1,1)
predictions[:5]
dfpredictions = pd.DataFrame(predictions, index=titanic_test_data.index)
dfpredictions.head(15)
dfpredictions = dfpredictions.rename(columns={0:'Survived'}).to_csv('submission.csv', header=True)