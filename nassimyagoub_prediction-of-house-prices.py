import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt
df = pd.read_csv("../input/kc_house_data.csv")

df = df.sample(frac=1)

df.head()
df["price"].mean()
df=df.drop(["id","zipcode"], axis = 1)
df.info()
import time

import datetime



def transfo(date):

    year = date[:4]

    month = date[4:6]

    day = date[6:8]

    

    s=day+"/"+month+"/"+year



    ans = time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple())

    

    return ans
df["date"]=df["date"].apply(transfo)
df["date"].head()
%matplotlib inline

df.hist(bins=20, figsize=(20,15))

plt.show()
df["price"].hist(bins=100, figsize=(10,5))

plt.show()
df["price_category"] = pd.cut(df["price"],

                            bins=list(range(0,2000001,100000))+[np.inf],

                            labels=list(range(21)))
df["price_category"].hist(figsize=(10,5))

plt.show()
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)

for train_index, test_index in split.split(df, df["price_category"]):

    train_set = df.loc[train_index]

    test_set = df.loc[test_index]
test_set["price_category"].value_counts() / len(test_set)
df["price_category"].value_counts() / len(df)
train_set=train_set.drop("price_category", axis = 1)

test_set=test_set.drop("price_category", axis = 1)
train_set.head()
prices = train_set.copy()
prices.plot(kind="scatter", x="long", y="lat", alpha = 0.1)
# We round those values in order to identify zones

prices["new_lat"]=round(prices["lat"],2)

prices["new_long"]=round(prices["long"],2)
# This is the indicator of a zone, each value for this column corresponds to a certain geographical zone

prices["zone"]=prices["new_lat"]*1000+prices["new_long"]
# The only columns we need for this representation

prices=prices[["zone","price", "lat", "long"]]

# This column will be used to count the number of houses in a zone

prices["number"]=1
# Will contain the average price for each zone

df=prices.groupby('zone').mean()

# Will contain the number of houses for each zone

df1=prices.groupby('zone').sum()

df["number"]=df1["number"]
df.plot(kind="scatter", x="long", y="lat", alpha=0.5,

        s=df["number"]*2, label="population",

        figsize=(15,10),c="price", cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend()

plt.savefig("prices",format="png",resolution=300)
from math import log

df["log_price"]=df["price"].apply(log)
df.plot(kind="scatter", x="long", y="lat", alpha=0.5,

        s=df["number"]*2, label="population",

        figsize=(15,10),c="log_price", cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend()

plt.savefig("log_prices",format="png",resolution=300)
corr_matrix = train_set.corr()

corr_matrix["price"].sort_values(ascending=False)
houses_data = train_set.drop("price", axis=1) # drop labels for training set

houses_labels = train_set["price"].copy()
test_data = test_set.drop("price", axis=1)

test_labels = test_set["price"].copy()
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

houses_prepared = std_scaler.fit_transform(houses_data)
houses_prepared.shape
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(houses_prepared, houses_labels)
from sklearn.metrics import mean_squared_error



houses_predictions = lin_reg.predict(houses_prepared)

lin_mse = mean_squared_error(houses_labels, houses_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor(random_state=1, max_depth = 12)

tree_reg.fit(houses_prepared, houses_labels)
houses_predictions = tree_reg.predict(houses_prepared)

tree_mse = mean_squared_error(houses_labels, houses_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
from sklearn.model_selection import cross_val_score



tree_scores = cross_val_score(tree_reg, houses_prepared, houses_labels,

                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-tree_scores)



print("Scores : ",tree_rmse_scores)

print("Mean : ",tree_rmse_scores.mean())

print("Standard deviation : ",tree_rmse_scores.std())
from sklearn.svm import SVR



svm_reg = SVR(gamma='scale')

svm_reg.fit(houses_prepared, houses_labels)
svm_scores = cross_val_score(svm_reg, houses_prepared, houses_labels,

                             scoring="neg_mean_squared_error", cv=10)



svm_rmse_scores = np.sqrt(-svm_scores)



print("Scores : ",svm_rmse_scores)

print("Mean : ",svm_rmse_scores.mean())

print("Standard deviation : ",svm_rmse_scores.std())
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=50, random_state=1)

forest_reg.fit(houses_prepared, houses_labels)
houses_predictions = forest_reg.predict(houses_prepared)

forest_mse = mean_squared_error(houses_labels, houses_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
forest_scores = cross_val_score(forest_reg, houses_prepared, houses_labels,

                                scoring="neg_mean_squared_error", cv=10)



forest_rmse_scores = np.sqrt(-forest_scores)



print("Scores : ",forest_rmse_scores)

print("Mean : ",forest_rmse_scores.mean())

print("Standard deviation : ",forest_rmse_scores.std())
from sklearn.model_selection import GridSearchCV



param_grid = [{"max_depth" : [10,15,20,30],

               "n_estimators" : [10, 50, 100]}]



grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(houses_prepared, houses_labels)
params = grid_search.best_params_

depth_param = params['max_depth']

estimator_param = params['n_estimators']
forest_reg = RandomForestRegressor(max_depth=depth_param, n_estimators=estimator_param, random_state=1)

forest_reg.fit(houses_prepared, houses_labels)
test_prepared = std_scaler.transform(test_data)

test_predict = forest_reg.predict(test_prepared)
forest_mse = mean_squared_error(test_labels, test_predict)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
print('The RMSE with this model is {} dollars on the validation dataset.'.format(forest_rmse))