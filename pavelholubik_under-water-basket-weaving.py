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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn import linear_model
from sklearn import preprocessing
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_train.head()
y_train = df_train["SalePrice"].values
df_train.drop("SalePrice", axis=1, inplace=True)
df_train.drop("Id", axis=1, inplace=True)
df_test.drop(["Id"], axis=1, inplace=True)
catCols = []

for col in df_train:
    if df_train[col].dtype == object or len(df_train[col].value_counts()) < 5:
        catCols.append(col)
    elif (df_train[col].dtype == int or df_train[col].dtype == float) and len(df_train[col].value_counts()) < 5:
        catCols.appent(col)
for col in catCols:
    df_train[col] = pd.factorize(df_train[col])[0]
    df_test[col] = pd.factorize(df_test[col])[0]
df_train.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)
df_train["LotConfig"].value_counts()
df_test["Street"].value_counts()
df_test.head()
rf = RandomForestClassifier(n_estimators=50)
rf.fit(df_train.values, y_train)
preds_rf = rf.predict(df_test.values)
boosting = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, 
                                     random_state=42, loss='ls')
boosting.fit(df_train.values, y_train)
preds_boost = boosting.predict(df_test.values)
br = linear_model.BayesianRidge()
br.fit(df_train.values, y_train)
preds_br = br.predict(df_test.values)
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(df_train.values, y_train)
preds_lasso = lasso.predict(df_test.values)
preds = np.mean([preds_br, preds_lasso, preds_rf, preds_boost], axis=0)
sub = pd.read_csv("../input/sample_submission.csv")
sub.head()
sub["SalePrice"] = preds
sub.head()
sub.to_csv("submission.csv", index=False)
