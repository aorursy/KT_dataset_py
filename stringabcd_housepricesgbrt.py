import numpy as np
import pandas as pd
df_test = pd.read_csv("../input/test.csv")
df_train = pd.read_csv("../input/train.csv")

df_train = df_train[df_train["GarageArea"] < 1200]

data = df_train.select_dtypes(include=[np.number]).interpolate().dropna()
y = np.log(df_train.SalePrice)
x = data.drop(["SalePrice", "Id"], axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
lModel = lr.fit(X_train, y_train)
lResult = lModel.predict(X_train)

X_train.loc[:,"LinearFeature"] = lResult

import xgboost as xgb
params = {'max_depth':4, 'colsample_bytree':0.8, 'subsample':0.8, 'eta':0.02, 'silent':1, 'eval_metric':'error', 'min_child_weight':2.5, 'seed':10}
dtrain = xgb.DMatrix(X_train, label=y_train)
bst = xgb.train(list(params.items()), dtrain, 900)
# X_test.loc[:, "LinearFeature"] = lModel.predict(X_test)
# dtest = xgb.DMatrix(X_test)
# final_predictions = bst.predict(dtest)
# from sklearn.metrics import mean_squared_error
# print (mean_squared_error(y_test, final_predictions))


feats = df_test.select_dtypes(include=[np.number]).drop(["Id"], axis=1).interpolate()
feats.loc[:, "LinearFeature"] = lModel.predict(feats)
dtest = xgb.DMatrix(feats)
final_predictions = bst.predict(dtest)
submission = pd.DataFrame()
submission["Id"] = df_test.Id
submission["SalePrice"] = np.exp(final_predictions)
submission.to_csv("submission3.csv", index=False)