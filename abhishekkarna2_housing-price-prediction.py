# This Python 3 environment comes with many helpful analytics libraries installedaaple
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
training_set = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
submission_set = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
training_set.drop(columns=["Id", "Alley", "MiscFeature", "Fence", "PoolQC", "FireplaceQu"], inplace=True)
submission_set.drop(columns=["Id", "Alley", "MiscFeature", "Fence", "PoolQC", "FireplaceQu"], inplace=True)
training_set.fillna(method ='bfill', inplace=True)
submission_set.fillna(method ='bfill', inplace=True)
submission_set.info()
training_set.info()
training_set.reset_index(drop=True, inplace=True)


print(training_set.Street.unique())
print(training_set.SaleCondition.unique())
print(training_set.LotShape.unique())
print(training_set.Utilities.unique())
print(training_set.LandContour.unique())
print(training_set.LotConfig.unique())
print(training_set.LandSlope.unique())
print(training_set.Neighborhood.unique())
print(training_set.Condition1.unique())
print(training_set.Condition2.unique())
print(training_set.BldgType.unique())
print(training_set.HouseStyle.unique())
print(training_set.RoofStyle.unique())
print(training_set.RoofMatl.unique())
print(training_set.Exterior1st.unique())
print(training_set.Exterior2nd.unique())
print(training_set.MasVnrType.unique())
print(training_set.ExterQual.unique())
print(training_set.ExterCond.unique())
print(training_set.Foundation.unique())
print(training_set.BsmtQual.unique())
print(training_set.BsmtCond.unique())
print(training_set.BsmtExposure.unique())
print(training_set.BsmtFinType1.unique())
print(training_set.BsmtFinType2.unique())
print(training_set.Heating.unique())
print(training_set.HeatingQC.unique())
print(training_set.CentralAir.unique())
print(training_set.Electrical.unique())
print(training_set.PavedDrive.unique())
print(training_set.KitchenQual.unique())
print(training_set.Functional.unique())
print(training_set.GarageType.unique())
print(training_set.GarageFinish.unique())
print(training_set.GarageQual.unique())
print(training_set.GarageCond.unique())
print(training_set.SaleType.unique())
print(training_set.MSZoning.unique())

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()
Label = le.fit_transform(training_set["Street"])
Label1 = le.fit_transform(training_set["SaleCondition"])
Label2 = le.fit_transform(training_set["LotShape"])
Label3 = le.fit_transform(training_set["LandContour"])
Label4 = le.fit_transform(training_set["LotConfig"])
Label5 = le.fit_transform(training_set["LandSlope"])
Label6 = le.fit_transform(training_set["Neighborhood"])
Label7 = le.fit_transform(training_set["Condition1"])
Label8 = le.fit_transform(training_set["Condition2"])
Label9 = le.fit_transform(training_set["BldgType"])
Label10 = le.fit_transform(training_set["HouseStyle"])
Label11 = le.fit_transform(training_set["RoofStyle"])
Label12 = le.fit_transform(training_set["RoofMatl"])
Label13 = le.fit_transform(training_set["ExterQual"])
Label14 = le.fit_transform(training_set["Foundation"])
Label15 = le.fit_transform(training_set["Heating"])
Label16 = le.fit_transform(training_set["HeatingQC"])
Label17 = le.fit_transform(training_set["CentralAir"])
Label18 = le.fit_transform(training_set["PavedDrive"])
Label19 = le.fit_transform(training_set["ExterCond"])
Label20 = le.fit_transform(training_set["BsmtQual"])
Label21 = le.fit_transform(training_set["Exterior1st"])
Label22 = le.fit_transform(training_set["Exterior2nd"])
Label23 = le.fit_transform(training_set["BsmtCond"])
Label24 = le.fit_transform(training_set["BsmtExposure"])
Label25 = le.fit_transform(training_set["BsmtFinType1"])
Label26 = le.fit_transform(training_set["BsmtFinType2"])
Label27 = le.fit_transform(training_set["Electrical"])
Label28 = le.fit_transform(training_set["KitchenQual"])
Label29 = le.fit_transform(training_set["Functional"])
Label30 = le.fit_transform(training_set["GarageType"])
Label31 = le.fit_transform(training_set["GarageQual"])
Label32 = le.fit_transform(training_set["GarageCond"])
Label33 = le.fit_transform(training_set["GarageFinish"])
Label34 = le.fit_transform(training_set["SaleType"])
Label35 = le.fit_transform(training_set["Utilities"])
Label36 = le.fit_transform(training_set["MasVnrType"])
Label37 = le.fit_transform(training_set["MSZoning"])


training_set.drop(columns=["Street", "SaleCondition", "LotShape", "LandContour", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "ExterQual", "ExterCond", "Foundation", "Heating", "HeatingQC", "CentralAir", "PavedDrive", "BsmtQual", "Exterior1st", "Exterior2nd", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Electrical", "KitchenQual", "Functional", "GarageType", "GarageQual", "GarageCond", "GarageFinish", "SaleType", "Utilities", "MasVnrType", "MSZoning"], inplace=True)
training_set["Street"] = Label
training_set["SaleCondition"] = Label1
training_set["LotShape"] = Label2
training_set["LandContour"]= Label3
training_set["LotConfig"]= Label4
training_set["LandSlope"]= Label5
training_set["Neighborhood"]= Label6
training_set["Condition1"]= Label7
training_set["Condition2"]= Label8
training_set["BldgType"]= Label9
training_set["HouseStyle"]= Label10
training_set["RoofStyle"]= Label11
training_set["RoofMatl"]= Label12
training_set["ExterQual"]= Label13
training_set["Foundation"]= Label14
training_set["Heating"]= Label15
training_set["HeatingQC"]= Label16
training_set["CentralAir"]= Label17
training_set["PavedDrive"]= Label18
training_set["ExterCond"]= Label19
training_set["BsmtQual"]= Label20
training_set["Exterior1st"]=Label21
training_set["Exterior2nd"]=Label22
training_set["BsmtCond"]=Label23
training_set["BsmtExposure"]=Label24
training_set["BsmtFinType1"]=Label25
training_set["BsmtFinType2"]=Label26
training_set["Electrical"]=Label27
training_set["KitchenQual"]=Label28
training_set["Functional"]=Label29
training_set["GarageType"]=Label30
training_set["GarageQual"]=Label31
training_set["GarageCond"]=Label32
training_set["GarageFinish"]=Label33
training_set["SaleType"]=Label34
training_set["Utilities"]=Label35
training_set["MasVnrType"]=Label36
training_set["MSZoning"]=Label37

Label = le.fit_transform(submission_set["Street"])
Label1 = le.fit_transform(submission_set["SaleCondition"])
Label2 = le.fit_transform(submission_set["LotShape"])
Label3 = le.fit_transform(submission_set["LandContour"])
Label4 = le.fit_transform(submission_set["LotConfig"])
Label5 = le.fit_transform(submission_set["LandSlope"])
Label6 = le.fit_transform(submission_set["Neighborhood"])
Label7 = le.fit_transform(submission_set["Condition1"])
Label8 = le.fit_transform(submission_set["Condition2"])
Label9 = le.fit_transform(submission_set["BldgType"])
Label10 = le.fit_transform(submission_set["HouseStyle"])
Label11 = le.fit_transform(submission_set["RoofStyle"])
Label12 = le.fit_transform(submission_set["RoofMatl"])
Label13 = le.fit_transform(submission_set["ExterQual"])
Label14 = le.fit_transform(submission_set["Foundation"])
Label15 = le.fit_transform(submission_set["Heating"])
Label16 = le.fit_transform(submission_set["HeatingQC"])
Label17 = le.fit_transform(submission_set["CentralAir"])
Label18 = le.fit_transform(submission_set["PavedDrive"])
Label19 = le.fit_transform(submission_set["ExterCond"])
Label20 = le.fit_transform(submission_set["BsmtQual"])
Label21 = le.fit_transform(submission_set["Exterior1st"])
Label22 = le.fit_transform(submission_set["Exterior2nd"])
Label23 = le.fit_transform(submission_set["BsmtCond"])
Label24 = le.fit_transform(submission_set["BsmtExposure"])
Label25 = le.fit_transform(submission_set["BsmtFinType1"])
Label26 = le.fit_transform(submission_set["BsmtFinType2"])
Label27 = le.fit_transform(submission_set["Electrical"])
Label28 = le.fit_transform(submission_set["KitchenQual"])
Label29 = le.fit_transform(submission_set["Functional"])
Label30 = le.fit_transform(submission_set["GarageType"])
Label31 = le.fit_transform(submission_set["GarageQual"])
Label32 = le.fit_transform(submission_set["GarageCond"])
Label33 = le.fit_transform(submission_set["GarageFinish"])
Label34 = le.fit_transform(submission_set["SaleType"])
Label35 = le.fit_transform(submission_set["Utilities"])
Label36 = le.fit_transform(submission_set["MasVnrType"])
Label37 = le.fit_transform(submission_set["MSZoning"])
submission_set.drop(columns=["Street", "SaleCondition", "LotShape", "LandContour", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "ExterQual", "ExterCond", "Foundation", "Heating", "HeatingQC", "CentralAir", "PavedDrive", "BsmtQual", "Exterior1st", "Exterior2nd", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Electrical", "KitchenQual", "Functional", "GarageType", "GarageQual", "GarageCond", "GarageFinish", "SaleType", "Utilities", "MasVnrType", "MSZoning"], inplace=True)
submission_set["Street"] = Label
submission_set["SaleCondition"] = Label1
submission_set["LotShape"] = Label2
submission_set["LandContour"]= Label3
submission_set["LotConfig"]= Label4
submission_set["LandSlope"]= Label5
submission_set["Neighborhood"]= Label6
submission_set["Condition1"]= Label7
submission_set["Condition2"]= Label8
submission_set["BldgType"]= Label9
submission_set["HouseStyle"]= Label10
submission_set["RoofStyle"]= Label11
submission_set["RoofMatl"]= Label12
submission_set["ExterQual"]= Label13
submission_set["Foundation"]= Label14
submission_set["Heating"]= Label15
submission_set["HeatingQC"]= Label16
submission_set["CentralAir"]= Label17
submission_set["PavedDrive"]= Label18
submission_set["ExterCond"]= Label19
submission_set["BsmtQual"]= Label20
submission_set["Exterior1st"]=Label21
submission_set["Exterior2nd"]=Label22
submission_set["BsmtCond"]=Label23
submission_set["BsmtExposure"]=Label24
submission_set["BsmtFinType1"]=Label25
submission_set["BsmtFinType2"]=Label26
submission_set["Electrical"]=Label27
submission_set["KitchenQual"]=Label28
submission_set["Functional"]=Label29
submission_set["GarageType"]=Label30
submission_set["GarageQual"]=Label31
submission_set["GarageCond"]=Label32
submission_set["GarageFinish"]=Label33
submission_set["SaleType"]=Label34
submission_set["Utilities"]=Label35
submission_set["MasVnrType"]=Label36
submission_set["MSZoning"]=Label37
label = training_set["SalePrice"]
train =training_set.drop(columns="SalePrice")
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(train)

scaled_data = scalar.transform(train)


scaled_data
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(scaled_data)


print(scaled_data.shape)

x_pca = pca.transform(scaled_data)

x_pca.shape
scalar = StandardScaler()
scalar.fit(submission_set)
scaled_data1 = scalar.transform(submission_set)



pca = PCA(n_components=5)
pca.fit(scaled_data1)


print(scaled_data1.shape)
x_pca1 = pca.transform(scaled_data1)
x_pca1.shape
data = pd.DataFrame(x_pca)
submit = pd.DataFrame(x_pca1)


x_train, x_test, y_train, y_test = train_test_split(data, label , test_size=0.2)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test)
import xgboost as xgb
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, learning_rate=0.1)
x_train
model.fit(x_train, y_train)
model.score(x_test, y_test)
submit.info()
predictions = model.predict(submit)
submission_set = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
submission = pd.DataFrame()
submission['Id'] = submission_set.Id
submission['SalePrice'] = predictions
submission.head()
submission.to_csv('submission1.csv', index=False)