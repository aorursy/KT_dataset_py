import pandas as pd

import numpy as np

from sklearn.linear_model import Lasso
train = pd.read_csv("../input/train.csv", na_values="NAN")

test = pd.read_csv("../input/test.csv", na_values="NAN")
targets = train['SalePrice']

train.drop('SalePrice', axis=1, inplace=True)
train.shape, test.shape
# trainとtestのデータ結合

all_data = pd.concat([train, test])

# カテゴリカルカラムをone-hot encodingへ変換

all_data = pd.get_dummies(all_data)

X = all_data.as_matrix()

#　NANを 0に

X = np.nan_to_num(X)
X.shape
#　train, validation, testデータに分割

X_train = X[:int(train.shape[0] * 0.8)]

y_train = targets[:int(train.shape[0] * 0.8)]



X_val = X[int(train.shape[0] * 0.8):train.shape[0]]

y_val = targets[int(train.shape[0] * 0.8):]



X_test = X[train.shape[0]:]
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape
clf = Lasso(alpha = 50)

clf.fit(X_train, y_train)
# https://www.kaggle.com/wiki/RootMeanSquaredError

Y = clf.predict(X_val)

from sklearn.metrics import mean_squared_error

RMSE = mean_squared_error(np.log(Y), np.log(y_val))**0.5 ; RMSE
Y = clf.predict(X_test)

out = pd.DataFrame()

out['Id'] = [i for i in range(train.shape[0]+1,train.shape[0]+test.shape[0]+1)]

out['SalePrice'] = Y

out.to_csv('output_lasso.csv', index=False)