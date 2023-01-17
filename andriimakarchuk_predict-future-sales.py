import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import f_regression
shopData = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

itemsData = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

categoriesData = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

salesTrain = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

testData = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

sampleData = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
print( sampleData )
print( testData )
dataSampleTest = testData.merge(sampleData, on="ID")

dataSampleTest.head(10)
data = salesTrain.merge( dataSampleTest, on=["shop_id", "item_id"] )

data.head()
data = data.drop( labels="date", axis=1 )

data.head()
data.astype("float64")

data.corr()
X = data.drop(labels=[ "item_cnt_day", "ID", "item_cnt_month" ], axis=1)

y1 = data["item_cnt_month"]

y2 = data["ID"]

regModelCnt = LinearRegression()

regModelID = LinearRegression()

regModelCnt.fit(X, y1)

regModelID.fit(X, y2)
print( "R^2 for 'item_cnt_month': "+str(regModelCnt.score(X, y1)) )

print( "R^2 for 'ID': "+str(regModelID.score(X, y2)) )
from sklearn import tree

regTreeCnt = tree.DecisionTreeRegressor()

regTreeID = tree.DecisionTreeRegressor()

regTreeCnt.fit(X, y1)

regTreeID.fit(X, y2)

print( "R^2 for 'item_cnt_month': "+str(regTreeCnt.score(X, y1)) )

print( "R^2 for 'ID': "+str(regTreeID.score(X, y2)) )