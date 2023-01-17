import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso, ElasticNet, Ridge

from sklearn import svm

from sklearn import neighbors

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
df = pd.read_csv('../input/london-bike-sharing-dataset/london_merged.csv')

df = df.drop(['timestamp'], 1)



print(df.shape)

print(df.head())
sns.pairplot(df)

plt.show()
prediction_label = 'wind_speed'
X = np.array(df.drop([prediction_label], 1))

y = np.array(df[prediction_label])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
estimator = Lasso(alpha=0.1)

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = ElasticNet()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = Ridge(alpha=0.5)

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = BaggingRegressor()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = RandomForestRegressor()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = ExtraTreesRegressor()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = AdaBoostRegressor()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = GradientBoostingRegressor(loss='ls')

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
print(df.head())
prediction_label = 'season'
X = np.array(df.drop([prediction_label], 1))

y = np.array(df[prediction_label])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
estimator = svm.SVC()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = svm.LinearSVC()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = neighbors.KNeighborsClassifier()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = neighbors.RadiusNeighborsClassifier(radius=1000)

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = BaggingClassifier()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = RandomForestClassifier()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = ExtraTreesClassifier()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = AdaBoostClassifier()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)
estimator = GradientBoostingClassifier()

estimator.fit(X_train, y_train)

accuracy = estimator.score(X_test, y_test)

print(accuracy)