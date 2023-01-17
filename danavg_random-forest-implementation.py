import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz

from sklearn import metrics

from sklearn.tree import _tree

import graphviz 



pd.options.display.max_rows = 20 # don't display many rows
train_data = pd.read_csv('../input/testtraincsv/train.csv')

test = pd.read_csv('../input/testtraincsv/test.csv')
train_data.describe().transpose()
X, y = train_data.drop('target', axis=1), train_data['target']
regr = RandomForestRegressor()

regr.fit(X, y)
print(f"R^2 : {regr.score(X, y)}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



regr = RandomForestRegressor()

regr.fit(X_train, y_train)

print(f"R^2 of train : {regr.score(X_train, y_train)}")

print(f"R^2 of test : {regr.score(X_test, y_test)}")
regr = RandomForestRegressor(n_estimators=40)

regr.fit(X_train, y_train)

print(f"R^2 of train: {regr.score(X_train, y_train)}")

print(f"R^2 of test: {regr.score(X_test, y_test)}")
regr = RandomForestRegressor(n_estimators=100)

regr.fit(X_train, y_train)

print(f"R-squared of train: {regr.score(X_train, y_train)}")

print(f"R-squared of test: {regr.score(X_test, y_test)}")
from sklearn.ensemble import GradientBoostingRegressor

gbrt=GradientBoostingRegressor(n_estimators=100)

gbrt.fit(X, y)

y_pred=gbrt.predict(test)
y_pred
index = test['id']

df = pd.DataFrame(y_pred, index=index)

df.columns = ['target']
df
df.to_csv("submit_me.csv")