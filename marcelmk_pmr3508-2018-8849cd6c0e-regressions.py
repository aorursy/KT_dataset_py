import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
train = pd.read_csv("../input/train.csv", index_col = 'Id')
train
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(38)
knn_score = cross_val_score(knn, train.drop('median_house_value', axis = 1), train['median_house_value'], cv = 10, scoring = 'r2')
knn_score.mean()
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv = 10)
lasso_score = cross_val_score(lasso, train.drop('median_house_value', axis = 1), train['median_house_value'], cv = 10, scoring = 'r2')
lasso_score.mean()
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(cv = 10)
ridge_score = cross_val_score(ridge, train.drop('median_house_value', axis = 1), train['median_house_value'], cv = 10, scoring = 'r2')
ridge_score.mean()