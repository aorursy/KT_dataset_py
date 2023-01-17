import pandas as pd
import sklearn
house = pd.read_csv("../input/train.csv",
      sep=r'\s*,\s*',
      engine='python',
      na_values="?")
house.shape
house.head()
import matplotlib.pyplot as plt
sample = house.sample(n=2000)
plt.scatter(sample["median_income"], sample["median_house_value"])
plt.xlabel('median_income')
plt.ylabel('median_house_value')
house['median_income'].corr(house['median_house_value'])
plt.scatter(sample["latitude"], sample["median_house_value"])
plt.xlabel('latitude')
plt.ylabel('median_house_value')
plt.scatter(sample["longitude"], sample["median_house_value"])
plt.xlabel('longitude')
plt.ylabel('median_house_value')
house['latitude'].corr(house['median_house_value'])
house['longitude'].corr(house['median_house_value'])
house['median_age'].corr(house['median_house_value'])
house['total_rooms'].corr(house['median_house_value'])
house['total_bedrooms'].corr(house['median_house_value'])
house['population'].corr(house['median_house_value'])
house['households'].corr(house['median_house_value'])
Xhouse = house[["median_income","total_rooms", "latitude"]]
Yhouse = house.median_house_value
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=200)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xhouse, Yhouse, cv=10)
scores
sum(scores) / len(scores)
houseTest = pd.read_csv("../input/test.csv",
      sep=r'\s*,\s*',
      engine='python',
      na_values="?")
knn.fit(Xhouse,Yhouse)
XhouseTest = houseTest[["median_income","total_rooms", "latitude"]]
YtestPred = knn.predict(XhouseTest)
Id = houseTest["Id"]
submission = pd.DataFrame({"Id": Id, "median_house_value": YtestPred})
submission.head()
submission.to_csv("submissionKNN.csv", index = False)
from sklearn import linear_model
Xhouse = house[["longitude", "latitude", "median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]]
Yhouse = house.median_house_value
XhouseTest = houseTest[["longitude", "latitude", "median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]]
ridge = linear_model.Ridge(alpha=1.0)
ridge.fit(Xhouse, Yhouse)
scores = cross_val_score(ridge, Xhouse, Yhouse, cv=10)
scores
sum(scores) / len(scores)
YtestPred = ridge.predict(XhouseTest)
Id = houseTest["Id"]
submission = pd.DataFrame({"Id": Id, "median_house_value": YtestPred})
num = submission._get_numeric_data()
num[num < 0] = -num
submission.head()
submission.to_csv("submissionRidge.csv", index = False)
from sklearn.tree import DecisionTreeRegressor
Xhouse = house[["longitude", "latitude", "median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]]
Yhouse = house.median_house_value
XhouseTest = houseTest[["longitude", "latitude", "median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]]
tree = DecisionTreeRegressor(max_depth=10)
scores = cross_val_score(tree, Xhouse, Yhouse, cv=10)
scores
sum(scores) / len(scores)
tree.fit(Xhouse, Yhouse)
YtestPred = tree.predict(XhouseTest)
Id = houseTest["Id"]
submission = pd.DataFrame({"Id": Id, "median_house_value": YtestPred})
submission.head()
submission.to_csv("submissionTree.csv", index = False)