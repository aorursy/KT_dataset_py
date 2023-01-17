# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from warnings import simplefilter

simplefilter("ignore")

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pyearth import Earth

from sklearn.preprocessing import scale
data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
for k, v in data.isna().mean().sort_values(ascending=False).to_dict().items():

    if v < 0.01:

        print(f'"{k}"', end=",")
cols = [

    "MasVnrArea","MasVnrType","Electrical","Utilities","YearRemodAdd","MSSubClass","Foundation","ExterCond","ExterQual","Exterior2nd","Exterior1st","RoofMatl","RoofStyle",

    "YearBuilt","LotConfig","OverallCond","OverallQual","HouseStyle","BldgType","Condition2","BsmtFinSF1","MSZoning","LotArea","Street","Condition1","Neighborhood","LotShape",

    "LandContour","LandSlope","HeatingQC","BsmtFinSF2","EnclosedPorch","Fireplaces","GarageCars","GarageArea","PavedDrive","WoodDeckSF","OpenPorchSF","3SsnPorch",

    "BsmtUnfSF","ScreenPorch","PoolArea","MiscVal","MoSold","YrSold","SaleType","Functional","TotRmsAbvGrd","KitchenQual","KitchenAbvGr","BedroomAbvGr",

    "HalfBath","FullBath","BsmtHalfBath","BsmtFullBath","GrLivArea","LowQualFinSF","2ndFlrSF","1stFlrSF","CentralAir","SaleCondition","Heating","TotalBsmtSF"

]

X = pd.concat([data[cols], test[cols]], ignore_index=True)

X_num = X._get_numeric_data()





X = pd.concat([

    pd.get_dummies(X.select_dtypes("object")),

    X_num.clip(X_num.quantile(0.01).to_dict(), X_num.quantile(0.99).to_dict(), axis=1)

], axis=1)



X = pd.DataFrame(scale(X), columns=X.columns).fillna(0)



X_train, X_test = X[:len(data)], X[len(data):]

y_train = data["SalePrice"].clip(data["SalePrice"].quantile(0.01), data["SalePrice"].quantile(0.99))

mu, sigma = np.log(y_train).median(), np.log(y_train).mad()

print(mu, sigma)

y_train = (np.log(y_train) - mu) / sigma



X_train.head()
from sklearn.feature_selection import RFE

m = RFE(Earth(), step=15, verbose=2).fit(X_train, y_train)
important_cols = X_train.columns[m.support_]
model = Earth().fit(X_train[important_cols], y_train)

y_test = model.predict(X_test[important_cols])

y_test = np.exp((y_test * sigma) + mu).astype(int)
result = pd.DataFrame({

    "Id": test.Id,

    "SalePrice": y_test

})



result.to_csv("submission.csv", index=False)