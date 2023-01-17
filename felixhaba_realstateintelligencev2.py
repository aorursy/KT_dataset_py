# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")
print ("Data loaded!")
test.head()
X = train.drop(["Id", "SalePrice"], axis = 1)
y = train.SalePrice
X_test = test.drop(["Id"], axis = 1)
features = pd.concat([X, X_test])
quantitative = [f for f in features.columns if features[f].dtype != "object"]
qualitative = [f for f in features.columns if features[f].dtype == "object"]
print(quantitative)

print(qualitative)
for q in qualitative:
    print (q, features[q].nunique())


train = train[train.GrLivArea < 4500]
outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])
missing = train.isnull().sum()
missing = missing[missing > 0]
print (missing)
#SimpleImputer
from sklearn.impute import SimpleImputer
#OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
#ColumnTransformer
from sklearn.compose import ColumnTransformer
#Pipeline
from sklearn.pipeline import Pipeline
#RandomForest
from sklearn.ensemble import RandomForestRegressor
#XGRBoost
from xgboost import XGBRegressor
#cross_val_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
numerical_transformer = SimpleImputer(strategy = "constant")

categorical_transformer = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy = "constant")),
    ("onehot", OneHotEncoder(handle_unknown = "ignore"))
])

preprocessor = ColumnTransformer(
transformers = [
    ("num", numerical_transformer, quantitative),
    ("cat", categorical_transformer, qualitative)
])

model = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=2, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)

pipeline = Pipeline(steps = [("preprocessor", preprocessor),
                            ("model", model)
                            ])
print("Cross validation")
print(cross_val_score(pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error'))

X_train, X_val, y_train, y_val = train_test_split(X, y)
pipeline.fit(X_train, y_train)
print("Train data")
print(pipeline.score(X_train, y_train))
print("Test data")
print(pipeline.score(X_val, y_val))


X.head()
X.shape
pipeline.fit(X,y)
predictions = pipeline.predict(X_test)
predictions = pd.DataFrame({'SalePrice': predictions}, index = test.Id)
predictions.to_csv("submission.csv")
predictions.head()