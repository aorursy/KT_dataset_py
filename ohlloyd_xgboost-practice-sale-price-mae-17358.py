import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
import xgboost as xgb
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
numeric = []
for i in train.columns:
    if train[i].dtypes != 'O':
        numeric.append(i)
df = train[numeric].dropna().drop(columns=['Id'])
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
def XGB_MAE(X, y, max_depth=3, lr=0.1, n_estimators=100, booster = "gbtree"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    mod = xgb.XGBRegressor(max_depth = max_depth, learning_rate = lr, n_estimators = n_estimators, booster = booster)
    mod = mod.fit(X_train, y_train)
    preds = mod.predict(X_test)
    mae = MAE(y_test, preds)
    return mae
best_score = 1e9
best_params = []
results = []


max_depths = [3,9,27,50,100,500]
lrs = [0.5, 0.1, 0.001, 0.0001]
n_estimators = [10, 50, 100, 300, 700]

for a in max_depths:
    for b in lrs:
        for c in n_estimators:
            mae = XGB_MAE(X, y, max_depth = a, lr = b, n_estimators = c)
            if mae < best_score:
                best_score = mae
                best_params = [a,b,c]
print("Best score was %f, achieved with depth %i, lr %f and estimators %i" % (best_score, best_params[0], best_params[1], best_params[2]))    
depth = best_params[0]
lrate = best_params[1]
estimators = best_params[2]
boosters = ['gbtree', 'gblinear', 'dart']

for boost in boosters:
    print(boost + " gave mae of: %f" % (XGB_MAE(X, y, max_depth = depth, lr = lrate, n_estimators = estimators, booster = boost)))