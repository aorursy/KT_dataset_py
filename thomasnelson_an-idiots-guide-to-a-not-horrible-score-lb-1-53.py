import pandas as pd
import numpy as np

dataset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')

X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values
X_Test = testset.iloc[:,1:].values
y = np.log(y)
from sklearn.feature_selection import VarianceThreshold
feature_selector = VarianceThreshold()
X = feature_selector.fit_transform(X)
X_Test = feature_selector.transform(X_Test)
from xgboost import XGBRegressor
regressor = XGBRegressor(n_estimators=300)
regressor.fit(X, y)
results = regressor.predict(X_Test)
results = np.exp(results)
submission = pd.DataFrame()
submission['ID'] = testset['ID']
submission['target'] = results
submission.to_csv('submission.csv', index=False)