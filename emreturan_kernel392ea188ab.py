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
data = pd.read_csv("../input/train.csv", index_col="Id")

test_data = pd.read_csv("../input/test.csv", index_col="Id")



cols = ["LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", 

        "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",

        "BsmtHalfBath", "FullBath", "HalfBath", "TotRmsAbvGrd", "Fireplaces",

        "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",

        "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"]
X = data[cols]

X = pd.get_dummies(X)

X = X.fillna(0)

y = data["SalePrice"]



test_X = test_data[cols]

test_X = pd.get_dummies(test_X)

test_X = test_X.fillna(0)
diff = np.setdiff1d(X.columns, test_X.columns)

for d in diff:

    test_X[d] = 0

test_X = test_X[list(X.columns)]
#from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler(feature_range=(0, 1))

#scaler = scaler.fit(X)

#X = pd.DataFrame(data=scaler.transform(X), index=X.index, columns=X.columns)

#test_X = pd.DataFrame(data=scaler.transform(test_X), index=test_X.index, columns=test_X.columns)
X

max(X["LotFrontage"])
from sklearn.linear_model import LinearRegression

regr = LinearRegression(normalize=True)

regr.fit(X, y)

pred = regr.predict(test_X)
output = pd.DataFrame(index=test_X.index)

output["SalePrice"] = pred
output.to_csv("output.csv")