# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import random
import ml_metrics as metrics


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.shape
test.shape
train.head(5)
test_ids = set(test.accuracy.unique())
train_ids = set(train.accuracy.unique())
intersection_count = len(test_ids & train_ids)
intersection_count == len(test_ids)
unique_places = train.place_id.unique()
#unique_rows = test.row_id.head(1000000)
#print(len(unique_places))

sel_place_ids = [unique_places[i] for i in sorted(random.sample(range(len(unique_places)), 100)) ]
sel_train = train[train.place_id.isin(sel_place_ids)]

#sel_row_ids = [unique_rows[i] for i in sorted(random.sample(range(len(unique_rows)), 1000000)) ]
#sel_test = test[test.row_id.isin(sel_row_ids)]

# Create the target and features numpy arrays: target, features_one
target = sel_train["place_id"].values
features_forest = sel_train[["x", "y"]].values
# Building and fitting my_forest
forest = RandomForestClassifier(n_estimators = 1000)
my_forest = forest.fit(features_forest, target)
print(my_forest.feature_importances_)
# Print the score of the fitted random forest
#print(my_forest.score(features_forest, target))
# Compute predictions on test set features then print the length of the prediction vector
test["place_id"] = float('NaN')
test.info()
test_features = test[["x", "y"]].values
print(test_features)
#pred_forest = forest.predict(test_features)
#print(len(pred_forest))
#row_id =np.array(test["row_id"]).astype(int)
#my_solution = pd.DataFrame(pred_forest, row_id, columns = ["place_id"])
#print(my_solution)
#my_solution.to_csv(fb_recruit, sep='\t')