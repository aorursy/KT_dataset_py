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
import pandas as pd

import numpy as np



data=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

data.info()

data=data.drop(columns=["BsmtExposure","BsmtFinType1","BsmtFinType2"])

data=data.drop(columns=["GarageType","GarageYrBlt","GarageFinish"])

data=data.drop(columns=["MiscFeature"])

data=data.drop(columns=["Id","LandSlope","Neighborhood","Condition1","Condition2","BldgType"])

data=data.drop(columns=["MSZoning","Heating","LotShape","LandContour","Utilities","LotConfig"])

data=data.drop(columns=["RoofStyle","RoofMatl","Exterior1st","Exterior2nd"])

data=data.drop(columns=["Functional","PavedDrive"])

data=data.drop(columns=["Foundation"])
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy="median")

data["LotFrontage"]=imputer.fit_transform(data[["LotFrontage"]])

data["MasVnrArea"]=imputer.fit_transform(data[["MasVnrArea"]])
data["Alley"] = data["Alley"].fillna("Mix")

data["MasVnrType"] = data["MasVnrType"].fillna("None")

data["BsmtQual"] = data["BsmtQual"].fillna("NA")

data["BsmtCond"] = data["BsmtCond"].fillna("NA")

data["Electrical"] = data["Electrical"].fillna("Mix")

data["FireplaceQu"] = data["FireplaceQu"].fillna("NA")

data["GarageQual"] = data["GarageQual"].fillna("NA")

data["GarageCond"] = data["GarageCond"].fillna("NA")

data["PoolQC"] = data["PoolQC"].fillna("TA")

data["Fence"] = data["Fence"].fillna("NA")
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

lbe=LabelEncoder()



'''data_lbl=data[["Alley","MasVnrType","BsmtQual","BsmtCond","Electrical","FireplaceQu","GarageQual","GarageCond","PoolQC",

             "Fence","HouseStyle","Street","ExterQual","ExterCond","HeatingQC","CentralAir","KitchenQual","SaleType","SaleCondition"]]

'''

data["Alley"]=lbe.fit_transform(data["Alley"])

data["MasVnrType"]=lbe.fit_transform(data["MasVnrType"])

data["BsmtQual"]=lbe.fit_transform(data["BsmtQual"])

data["BsmtCond"]=lbe.fit_transform(data["BsmtCond"])

data["Electrical"]=lbe.fit_transform(data["Electrical"])

data["FireplaceQu"]=lbe.fit_transform(data["FireplaceQu"])

data["GarageQual"]=lbe.fit_transform(data["GarageQual"])

data["GarageCond"]=lbe.fit_transform(data["GarageCond"])

data["PoolQC"]=lbe.fit_transform(data["PoolQC"])

data["Fence"]=lbe.fit_transform(data["Fence"])

data["HouseStyle"]=lbe.fit_transform(data["HouseStyle"])

data["Street"]=lbe.fit_transform(data["Street"])

data["ExterQual"]=lbe.fit_transform(data["ExterQual"])

data["ExterCond"]=lbe.fit_transform(data["ExterCond"])

data["HeatingQC"]=lbe.fit_transform(data["HeatingQC"])

data["CentralAir"]=lbe.fit_transform(data["CentralAir"])

data["KitchenQual"]=lbe.fit_transform(data["KitchenQual"])

data["SaleType"]=lbe.fit_transform(data["SaleType"])

data["SaleCondition"]=lbe.fit_transform(data["SaleCondition"])
'''

onehotencoder = OneHotEncoder(categorical_features ="all",categories='auto',handle_unknown='error')

data= onehotencoder.fit_transform(data).toarray()

'''
data.head()
X=data.iloc[:,:-1]

y=data.iloc[:,-1:]

X.shape,y.shape
import matplotlib.pyplot as plt

import seaborn as sns
correlations=data.corr()["SalePrice"].sort_values(ascending=False)

correlations
sns.regplot(data["OverallQual"],data["SalePrice"],data=data)
sns.regplot(data["GrLivArea"],data["SalePrice"],data=data)
sns.regplot(data["ExterQual"],data["SalePrice"],data=data)


sns.boxplot(data["Electrical"],data["SalePrice"],data=data)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()



lr.fit(X,y)

y_pred=lr.predict(X)

y_pred=y_pred.round()



from sklearn.metrics import mean_squared_error

print("RMSE: %.2f"%np.sqrt(mean_squared_error(y,y_pred)))



data["Pediction"]=y_pred
from sklearn.tree import DecisionTreeClassifier

trees=DecisionTreeClassifier()



trees.fit(X,y)

y_pred=trees.predict(X)





from sklearn.metrics import mean_squared_error

print("MSE: %.2f"%mean_squared_error(y,y_pred))

print("RMSE: %.2f"%np.sqrt(mean_squared_error(y,y_pred)))

from sklearn.ensemble import BaggingClassifier

bag=BaggingClassifier(DecisionTreeClassifier())



bag.fit(X,y)

y_pred=bag.predict(X)

y_pred=y_pred.round()



from sklearn.metrics import mean_squared_error

print("RMSE: %.2f"%np.sqrt(mean_squared_error(y,y_pred)))
from sklearn.ensemble import RandomForestClassifier

rad=RandomForestClassifier()



rad.fit(X,y)

y_pred=rad.predict(X)

y_pred=y_pred.round()



from sklearn.metrics import mean_squared_error

print("RMSE: %.2f"%np.sqrt(mean_squared_error(y,y_pred)))
from sklearn.linear_model import ElasticNet

en=ElasticNet(alpha=1)



en.fit(X,y)

y_pred=en.predict(X)

y_pred=y_pred.round()



from sklearn.metrics import mean_squared_error

print("RMSE: %.2f"%np.sqrt(mean_squared_error(y,y_pred)))
data_t=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

data_t.head()
data_t=data_t.drop(columns=["BsmtExposure","BsmtFinType1","BsmtFinType2"])

data_t=data_t.drop(columns=["GarageType","GarageYrBlt","GarageFinish"])

data_t=data_t.drop(columns=["MiscFeature","Foundation","Functional","PavedDrive"])

data_t=data_t.drop(columns=["Id","LandSlope","Neighborhood","Condition1","Condition2","BldgType"])

data_t=data_t.drop(columns=["MSZoning","Heating","LotShape","LandContour","Utilities","LotConfig"])

data_t=data_t.drop(columns=["RoofStyle","RoofMatl","Exterior1st","Exterior2nd"])
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy="median")

data_t["LotFrontage"]=imputer.fit_transform(data_t[["LotFrontage"]])

data_t["MasVnrArea"]=imputer.fit_transform(data_t[["MasVnrArea"]])



data_t["BsmtFinSF1"]=imputer.fit_transform(data_t[["BsmtFinSF1"]])

data_t["BsmtFinSF2"]=imputer.fit_transform(data_t[["BsmtFinSF2"]])

data_t["BsmtUnfSF"]=imputer.fit_transform(data_t[["BsmtUnfSF"]])

data_t["TotalBsmtSF"]=imputer.fit_transform(data_t[["TotalBsmtSF"]])



data_t["BsmtFullBath"]=imputer.fit_transform(data_t[["BsmtFullBath"]])

data_t["BsmtHalfBath"]=imputer.fit_transform(data_t[["BsmtHalfBath"]])

data_t["GarageCars"]=imputer.fit_transform(data_t[["GarageCars"]])

data_t["GarageArea"]=imputer.fit_transform(data_t[["GarageArea"]])

data_t["Alley"] = data_t["Alley"].fillna("Mix")

data_t["MasVnrType"] = data_t["MasVnrType"].fillna("None")

data_t["BsmtQual"] = data_t["BsmtQual"].fillna("NA")

data_t["BsmtCond"] = data_t["BsmtCond"].fillna("NA")

data_t["Electrical"] = data_t["Electrical"].fillna("Mix")

data_t["FireplaceQu"] = data_t["FireplaceQu"].fillna("NA")

data_t["GarageQual"] = data_t["GarageQual"].fillna("NA")

data_t["GarageCond"] = data_t["GarageCond"].fillna("NA")

data_t["PoolQC"] = data_t["PoolQC"].fillna("TA")

data_t["Fence"] = data_t["Fence"].fillna("NA")

data_t["KitchenQual"] = data_t["KitchenQual"].fillna("TA")

data_t["SaleType"] = data_t["SaleType"].fillna("New")

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

lbe=LabelEncoder()

data_t["Alley"]=lbe.fit_transform(data_t["Alley"])

data_t["MasVnrType"]=lbe.fit_transform(data_t["MasVnrType"])

data_t["BsmtQual"]=lbe.fit_transform(data_t["BsmtQual"])

data_t["BsmtCond"]=lbe.fit_transform(data_t["BsmtCond"])

data_t["Electrical"]=lbe.fit_transform(data_t["Electrical"])

data_t["FireplaceQu"]=lbe.fit_transform(data_t["FireplaceQu"])

data_t["GarageQual"]=lbe.fit_transform(data_t["GarageQual"])

data_t["GarageCond"]=lbe.fit_transform(data_t["GarageCond"])

data_t["PoolQC"]=lbe.fit_transform(data_t["PoolQC"])

data_t["Fence"]=lbe.fit_transform(data_t["Fence"])

data_t["HouseStyle"]=lbe.fit_transform(data_t["HouseStyle"])

data_t["Street"]=lbe.fit_transform(data_t["Street"])

data_t["ExterQual"]=lbe.fit_transform(data_t["ExterQual"])

data_t["ExterCond"]=lbe.fit_transform(data_t["ExterCond"])

data_t["HeatingQC"]=lbe.fit_transform(data_t["HeatingQC"])

data_t["CentralAir"]=lbe.fit_transform(data_t["CentralAir"])

data_t["KitchenQual"]=lbe.fit_transform(data_t["KitchenQual"])

data_t["SaleType"]=lbe.fit_transform(data_t["SaleType"])

data_t["SaleCondition"]=lbe.fit_transform(data_t["SaleCondition"])

from sklearn.tree import DecisionTreeClassifier

trees=DecisionTreeClassifier()



trees.fit(X,y)



data_pre=trees.predict(data_t)

data_t["Pred"]=data_pred

data_t