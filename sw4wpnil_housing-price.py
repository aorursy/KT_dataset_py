# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option("display.max_columns", None)

pd.set_option("display.max_row", None)
train_filepath = "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"

test_filepath = "/kaggle/input/house-prices-advanced-regression-techniques/test.csv"

submission_filepath = "/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv"

description_filepath = "/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt"
train_data = pd.read_csv(train_filepath)

test_data = pd.read_csv(test_filepath)

submission_data = pd.read_csv(submission_filepath)
train_data.head()
m_train = pd.concat([train_data.isna().sum(),train_data.isna().sum()/train_data.isna().count()], keys=["Total", "Percent"], axis=1) 

m_train[m_train["Percent"]>0.7]
train_data.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1, inplace=True)

test_data.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1, inplace=True)
# train_prof = ProfileReport(train_data)

# train_prof.to_file(output_file='./train_output.html')



# test_prof = ProfileReport(test_data)

# test_prof.to_file(output_file='./test_output.html')
train_data.drop(["MiscVal", "BsmtFinSF2", "LowQualFinSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "Street", "Utilities", "Condition2", "RoofMatl"], axis=1, inplace=True)

test_data.drop(["MiscVal", "BsmtFinSF2", "LowQualFinSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "Street", "Utilities", "Condition2", "RoofMatl"], axis=1, inplace=True)
train_data.describe()
train_obj = train_data.select_dtypes(include=object)

train_int = train_data.select_dtypes(exclude=object)
train_int.isnull().sum()
train_obj.isnull().sum()
train_data.LotFrontage = train_data.LotFrontage.fillna(train_data.LotFrontage.mean())



# train_data.MasVnrArea[train_data["MasVnrArea"]>0].mean() filling this value

train_data.MasVnrArea = train_data.MasVnrArea.fillna(254.0)



# train_data.GarageYrBlt.mode()

train_data.GarageYrBlt = train_data.GarageYrBlt.fillna(2005)

train_data.BsmtQual = train_data.BsmtQual.fillna("TA")

train_data.BsmtCond = train_data.BsmtCond.fillna("TA")

train_data.BsmtExposure = train_data.BsmtExposure.fillna("No")

train_data.BsmtFinType1 = train_data.BsmtFinType1.fillna("Unf")

train_data.BsmtFinType2 = train_data.BsmtFinType2.fillna("Unf")



train_data.Electrical = train_data.Electrical.fillna("SBrkr")



train_data.FireplaceQu = train_data.FireplaceQu.fillna("Gd")



train_data.GarageType = train_data.GarageType.fillna("Attchd")

train_data.GarageFinish = train_data.GarageFinish.fillna("Unf")

train_data.GarageQual = train_data.GarageQual.fillna("TA")

train_data.GarageCond = train_data.GarageCond.fillna("TA")







test_obj = test_data.select_dtypes(include=object)

test_int = test_data.select_dtypes(exclude=object)
test_int.isnull().sum()
test_obj.isnull().sum()
test_data.LotFrontage = test_data.LotFrontage.fillna(test_data.LotFrontage.mean())

test_data.MasVnrArea = test_data.MasVnrArea.fillna(test_data.MasVnrArea.mean())

test_data.MasVnrArea = test_data.MasVnrArea.fillna(test_data.MasVnrArea.mean())

test_data.BsmtFinSF1 = test_data.BsmtFinSF1.fillna(test_data.BsmtFinSF1.mean())

test_data.BsmtUnfSF = test_data.BsmtUnfSF.fillna(test_data.BsmtUnfSF.mean())

test_data.TotalBsmtSF = test_data.TotalBsmtSF.fillna(test_data.TotalBsmtSF.mean())



# test_data.BsmtFullBath.mode()

test_data.BsmtFullBath = test_data.BsmtFullBath.fillna(0.0)



# test_data.BsmtHalfBath.mode()

test_data.BsmtHalfBath = test_data.BsmtHalfBath.fillna(0.0)



# test_data.GarageYrBlt.mode()

test_data.GarageYrBlt = test_data.GarageYrBlt.fillna(2005.0)



# test_data.GarageCars.mode()

test_data.GarageCars = test_data.GarageCars.fillna(2.0)



# test_data.GarageArea.mode()

test_data.GarageArea = test_data.GarageArea.fillna(test_data.GarageArea.mean())
test_data.MSZoning = test_data.MSZoning.fillna("RL")

test_data.MasVnrType = test_data.MasVnrType.fillna("None")

test_data.BsmtQual = test_data.BsmtQual.fillna("TA")

test_data.BsmtCond = test_data.BsmtCond.fillna("TA")

test_data.BsmtExposure = test_data.BsmtExposure.fillna("No")

test_data.BsmtFinType1 = test_data.BsmtFinType1.fillna("GLQ")

test_data.BsmtFinType2 = test_data.BsmtFinType2.fillna("Unf")

test_data.KitchenQual = test_data.KitchenQual.fillna("TA")

test_data.Functional = test_data.Functional.fillna("Typ")

test_data.GarageFinish = test_data.GarageFinish.fillna("Unf")

test_data.GarageQual = test_data.GarageQual.fillna("TA")

test_data.GarageCond = test_data.GarageCond.fillna("TA")

test_data.FireplaceQu = test_data.FireplaceQu.fillna("Gd")

test_data.GarageType = test_data.GarageType.fillna("Attchd")

test_data.SaleType = test_data.SaleType.fillna("WD")

test_data.Exterior1st = test_data.Exterior1st.fillna("VinylSd")

test_data.Exterior2nd = test_data.Exterior2nd.fillna("VinylSd")
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()
for col in train_data.select_dtypes(include=object).columns:

    train_data[col] = le.fit_transform(train_data[col].astype(str))

X = train_data.drop(["Id", "SalePrice"], axis=1)

y = train_data["SalePrice"]
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, scorer







from sklearn.model_selection import train_test_split

lr = LinearRegression()

rf = RandomForestRegressor()



X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
for max_leaf_nodes in [5, 50, 500, 5000]:

    my_mae = get_mae(max_leaf_nodes, X_train, x_test, Y_train, y_test)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
X_train.shape, Y_train.shape, x_test.shape, y_test.shape
dtr = DecisionTreeRegressor(max_leaf_nodes=50, random_state=1)
lr.fit(X_train, Y_train)

dtr.fit(X_train, Y_train)

rf.fit(X_train, Y_train)
y_pred_lr = lr.predict(x_test)

y_pred_dtr = dtr.predict(x_test)

y_pred_rf = rf.predict(x_test)


a_lr = mean_absolute_error(y_pred_lr, y_test)

a_dtr = mean_absolute_error(y_pred_dtr, y_test)

a_rf = mean_absolute_error(y_pred_rf, y_test)



b_lr = mean_squared_error(y_pred_lr, y_test)

b_dtr = mean_squared_error(y_pred_dtr, y_test)

b_rf = mean_squared_error(y_pred_rf, y_test)



a_lr, b_lr, a_dtr, b_dtr, a_rf, b_rf

rf_model_on_full_data = RandomForestRegressor()



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)
test_x = test_data.drop("Id", axis=1)

test_y = submission_data.SalePrice
for col in test_x.select_dtypes(include=object).columns:

    test_x[col] = le.fit_transform(test_x[col].astype(str))
test_pred = rf_model_on_full_data.predict(test_x)
output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_pred})

output.to_csv('submission.csv', index=False)
a = mean_absolute_error(test_pred, test_y)

b = mean_squared_error(test_pred, test_y)

a, b