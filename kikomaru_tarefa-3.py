import pandas as pd
import numpy
data = pd.read_csv("../input/datasetss/train.csv")
data
data = data.drop('Id', axis=1)
data.isna().sum()
X_train = data.drop('median_house_value', axis=1)
y_train = data.median_house_value
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
lasso = linear_model.Lasso(alpha=0.1)
score1 = cross_val_score(lasso, X_train, y_train, cv=10)
score1.mean()
ridge = linear_model.Ridge(alpha=0.1)
score2 = cross_val_score(ridge, X_train, y_train, cv=10)
score2.mean()
from sklearn.neighbors import KNeighborsRegressor
score3 = []
neighbor = []
for k in range(10, 100, 10):
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    score3.append(cross_val_score(knn, X_train, y_train, cv=10).mean())
    neighbor.append(k)
score3
import matplotlib.pyplot as plt
plt.plot(neighbor, score3, 'ro')
score3 = []
neighbor = []
for k in range(1, 32, 2):
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    score3.append(cross_val_score(knn, X_train, y_train, cv=10).mean())
    neighbor.append(k)
plt.plot(neighbor, score3, 'ro')
neighbor
score3
knn = KNeighborsRegressor(n_neighbors=21)
score3 = cross_val_score(knn, X_train, y_train, cv=10)
score3.mean()
from sklearn.ensemble import RandomForestRegressor
score4 = []
trees = []
for n in range(10,101,10):
    regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=n)
    score4.append(cross_val_score(regr, X_train, y_train, cv=10).mean())
    trees.append(n)
plt.plot(trees, score4, 'ro')
regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=80)
score4.append(cross_val_score(regr, X_train, y_train, cv=10).mean())
score4[-1]
test = pd.read_csv('../input/datasetss/test.csv')
test.head()
X_test = test.drop('Id',axis=1)
lasso.fit(X_train, y_train)
prediction1 = lasso.predict(X_test)
prediction1 = abs(prediction)
prediction1[0]
prediction = pd.DataFrame({'Id':test.Id,'median_house_value': prediction1[0]})
prediction
prediction.to_csv('submition.csv', index = False)