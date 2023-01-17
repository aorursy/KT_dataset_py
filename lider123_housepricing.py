# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import math
import seaborn as sns

sns.set()
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.head(3)
train_data.info()
features = ["MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street", "Alley",
            "LotShape", "LandContour", "LotConfig", "LandSlope", "Condition1",
            "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond",
            "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st",
            "Exterior2nd", "MasVnrType", "MasVnrArea", "ExterQual", "ExterCond",
            "Foundation", "Heating", "CentralAir", "Electrical", "GrLivArea",
            "GarageArea", "PoolArea", "MiscVal", "YrSold", "SaleType",
            "SaleCondition"]

y_train = train_data["SalePrice"]
train_size = len(train_data)
X = pd.concat([train_data[features], test_data[features]])
X.info()
X["SaleType"].fillna(X["SaleType"].mode()[0], inplace=True)
X["Electrical"].fillna(X["Electrical"].mode()[0], inplace=True)
X["MSZoning"].fillna(X["MSZoning"].mode()[0], inplace=True)
X["GarageArea"].fillna(X["GarageArea"].median(), inplace=True)
X["CentralAir"] = X["CentralAir"].map({"Y": 1, "N": 0})
X["LotFrontage"].fillna(X["LotFrontage"].median(), inplace=True)
X["Alley"].fillna("None", inplace=True)
X["MasVnrType"].fillna("None", inplace=True)
X["MasVnrArea"].fillna(0, inplace=True)

categorical_features = ["MSSubClass", "MSZoning", "Street", "Alley", "LotShape",
                        "LandContour", "LotConfig", "LandSlope", "Condition1",
                        "Condition2", "BldgType", "HouseStyle", "OverallQual",
                        "OverallCond", "RoofStyle", "RoofMatl", "Exterior1st",
                        "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond",
                        "Foundation", "Heating", "Electrical", "SaleType",
                        "SaleCondition"]
X = pd.get_dummies(X, columns=categorical_features)

X_train, X_test = X.iloc[:train_size, :], X.iloc[train_size:, :]

X_train.head()
params = {
    "n_estimators": [10, 20, 30, 50, 100, 150, 200],
    "max_features": ["auto", "sqrt", "log2", None],
    "max_depth": list(range(2, 11)) + [None]
}
gs = GridSearchCV(estimator=RandomForestRegressor(), param_grid=params, cv=5, scoring="neg_mean_squared_error")
gs.fit(X_train, y_train)
gs.best_params_
model = RandomForestRegressor(**gs.best_params_)
model.fit(X_train, y_train)
def evaluate(y, y_pred):
    y = list(map(math.log, y))
    y_pred = list(map(math.log, y_pred))
    return math.sqrt(mean_squared_error(y, y_pred))

scorer = make_scorer(evaluate)
print("Mean score:", cross_val_score(estimator=model, X=X_train, y=y_train, cv=5, scoring=scorer).mean())
test_data[["Id"]].assign(SalePrice=model.predict(X_test)).to_csv("prediction.csv", index=False)
