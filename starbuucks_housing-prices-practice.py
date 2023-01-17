import pandas as pd



train = pd.read_csv('../input/home-data-for-ml-course/train.csv')

test = pd.read_csv('../input/home-data-for-ml-course/test.csv')
train.columns
y = train['SalePrice']



features = ['LotArea', 'YearBuilt', 'GarageYrBlt', 'PoolArea', 'GarageCars']

X = train[features]
X['GarageYrBlt'] = X['GarageYrBlt'].fillna(1978)
from sklearn.tree import DecisionTreeRegressor



model = DecisionTreeRegressor()

model.fit(X, y)
X_test = test[features]
X_test['GarageYrBlt'] = X_test['GarageYrBlt'].fillna(X_test['GarageYrBlt'].mean())

X_test['GarageCars'] = X_test['GarageCars'].fillna(X_test['GarageCars'].mean())
preds = model.predict(X_test)
submission = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')

submission.head()
submission['SalePrice'] = preds
submission.to_csv('submission_1.csv', index=False)