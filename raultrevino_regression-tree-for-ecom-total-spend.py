import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
data = pd.read_csv("../input/Ecom Expense.csv")

data.head()
data.shape
print(data.dtypes)
data = data.drop(['Transaction ID'], axis=1)
print("Numero de registros:"+str(data.shape[0]))

for column in data.columns.values:

    print(column + "-NAs:"+ str(pd.isnull(data[column]).values.ravel().sum()))
data.head()
dummy_gender = pd.get_dummies(data["Gender"],prefix="Gender")

dummy_city_tier = pd.get_dummies(data["City Tier"],prefix="City")
data = data.drop(['Gender'], axis=1)

data = data.drop(['City Tier'], axis=1)
data = data[data.columns.values.tolist()].join(dummy_gender)

data = data[data.columns.values.tolist()].join(dummy_city_tier)

data.head()
data_vars = data.columns.values.tolist()

Y = ['Total Spend']

X = [v for v in data_vars if v not in Y]

X_train, X_test, Y_train, Y_test = train_test_split(data[X],data[Y],test_size = 0.3, random_state=0)

Y_test.head()
regtree = DecisionTreeRegressor(min_samples_split=2,min_samples_leaf=1,random_state=0)

regtree.fit(X_train,Y_train)
preds = regtree.predict(X_test[X])
test_preds = pd.DataFrame(preds,columns= ['tree_preds'])
test_preds['real_value'] = Y_test.values.tolist()
test_preds.head()
score = regtree.score(X_test,Y_test)

score
test_preds.tail()
list(zip(X,regtree.feature_importances_))
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_jobs=2,oob_score=True,n_estimators=500)

forest.fit(X_train,Y_train.values.ravel()) #n_estimators son el numero de arboles 
forest_test_pred = forest.predict(X_test)
forest_test_preds = pd.DataFrame(forest_test_pred,columns= ['forest_preds'])

forest_test_preds["real_pred"] =  Y_test.values.ravel()

forest_test_preds.head()
forest.oob_score_  
forest_test_preds["rforest_error2"] = (forest_test_preds["forest_preds"] - forest_test_preds["real_pred"] )**2

sum(forest_test_preds["rforest_error2"])/len(Y_test)