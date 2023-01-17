import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import Imputer

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from ml_metrics import rmse
train = pd.read_csv("../input/train.csv").select_dtypes(exclude=['object'])

test = pd.read_csv("../input/test.csv").select_dtypes(exclude=['object'])
x = train.drop("SalePrice", axis=1)

i = Imputer()

x = i.fit_transform(x)

y = train["SalePrice"]

np.random.seed(1337)

xtrain, xtest, ytrain, ytest = train_test_split(x,y)

model = RandomForestRegressor()
model.fit(xtrain, ytrain)
pred = model.predict(xtest)

rmse(np.log(ytest), np.log(pred))
model.fit(x, y)
preds = model.predict(i.transform(test))

sub = pd.DataFrame({'Id': test.Id, 'SalePrice': preds})

sub.to_csv("submission.csv", index=False)