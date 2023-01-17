# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))        
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
print('Shape of the taining set') 
print(train.shape)
print('Information about the training set\n') 
train.info()
train.describe()
# Let's take the statistical summary of "Object" data

train.describe(include=['O'])
sns.distplot(train.SalePrice)

train.plot(kind = 'scatter',x="Id", y="SalePrice", color = 'r',label = 'Price',
           linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
train_inlier = train[train.SalePrice < 400000]
train_inlier.shape
train_saleprice= train.SalePrice
train_inlier = train_inlier.drop(labels=["SalePrice", "Id"], axis=1)

test_id = test.Id
test = test.drop(labels=["Id"], axis=1)
df = pd.concat([train_inlier, test])
df.isnull().sum().sort_values(ascending=False).head(15)
df.drop(labels = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
                     "LotFrontage"], axis = 1, inplace = True)
#evin 2. katın m2'sini gösteriyor ama 2. katı olmayan evler 0 olarak yazılmış.
#0 olan değerlere bakılıp fazla ise bu kolon silinebilir.

print(df["2ndFlrSF"].value_counts()) #1658 değer 0 çıktı. Bu yarısından fazlası demektir.
df.drop("2ndFlrSF", axis=1, inplace = True)
df.drop(labels = ["WoodDeckSF", "OpenPorchSF","EnclosedPorch","3SsnPorch","PoolArea",
                  "MasVnrArea","BsmtFinSF1","BsmtFinSF2","LowQualFinSF",
                  "ScreenPorch", "MiscVal"], axis = 1, inplace = True)

df.shape
missingValue = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
missingValue = missingValue.fit(df.iloc[:, 0:61])
df.iloc[:, 0:61] = missingValue.transform(df.iloc[:, 0:61])
missingValue = SimpleImputer(missing_values = 0, strategy = 'mean')
missingValue = missingValue.fit(df.iloc[:,31:33])
df.iloc[:, 31:33] = missingValue.transform(df.iloc[:, 31:33])
missingValue = SimpleImputer(missing_values = 0, strategy = 'mean')
missingValue = missingValue.fit(df.iloc[:,53:54])
df.iloc[:, 53:54] = missingValue.transform(df.iloc[:, 53:54])
df.isna().sum().sum()
df.loc[:,["MSSubClass","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFullBath",
             "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
             "Fireplaces", "GarageCars", "GarageYrBlt", "MoSold",
             "YrSold"]] = df.loc[:,["MSSubClass", "OverallQual", "OverallCond", "YearBuilt",
                                       "YearRemodAdd", "BsmtFullBath", "BsmtHalfBath", "FullBath",
                                       "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
                                       "Fireplaces", "GarageCars", "GarageYrBlt", "MoSold",
                                       "YrSold"]].astype("object")
df_obj = df.describe(include=["O"])
df_obj = df_obj.columns
df = pd.get_dummies(df, columns = df_obj, drop_first = True)

df.shape
train_dumm = df[:1432]
test_dumm = df[1432:]
train_dumm["SalePrice"] = train_saleprice
cor = train_dumm.corr().abs()["SalePrice"]
cor[cor>0.5]
train_corr = train_dumm.loc[:,["TotalBsmtSF", "1stFlrSF", "GrLivArea", "GarageArea", "ExterQual_Gd",
                               "ExterQual_TA", "Foundation_PConc", "BsmtQual_TA", "FullBath_1",
                               "FullBath_2", "KitchenQual_TA", "GarageFinish_Unf", "GarageCars_3.0","SalePrice"]]

X_test = test_dumm.loc[:,["TotalBsmtSF", "1stFlrSF", "GrLivArea", "GarageArea", "ExterQual_Gd",
                               "ExterQual_TA", "Foundation_PConc", "BsmtQual_TA", "FullBath_1",
                               "FullBath_2", "KitchenQual_TA", "GarageFinish_Unf", "GarageCars_3.0"]]
X = train_corr.drop("SalePrice", axis=1).values
y = train_corr.loc[:, "SalePrice"].values
makine = LinearRegression()
makine.fit(X, y)
y_pred = makine.predict(X)
print("Root mean square error train = " + str(np.sqrt(mean_squared_error(y, y_pred))))
print("R2 score = " + str(r2_score(y, y_pred)))
y_test = makine.predict(X_test)

import statsmodels.api as sm
# Modele sabit terim (b0) için birlerden oluşan bir sütun ekleyelim

X_bs = np.append(arr = X, values=np.ones((1432, 1)).astype(int), axis=1)

X_bs
X_bs.shape
X_gds = X_bs[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]

anlamlilik_duzeyi = 0.05
regresyon_gds = sm.OLS(endog=y, exog=X_gds).fit()
regresyon_gds.summary()
X_gds = X_bs[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]]

regresyon_gds = sm.OLS(endog=y, exog=X_gds).fit()
regresyon_gds.summary()
X_gds = X_bs[:, [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13]]

regresyon_gds = sm.OLS(endog=y, exog=X_gds).fit()
regresyon_gds.summary()
X_gds = X_bs[:, [0, 1, 2, 3, 5, 6, 7, 10, 11, 12, 13]]
regresyon_gds = sm.OLS(endog=y, exog=X_gds).fit()
regresyon_gds.summary()
X_gds = X_bs[:, [0, 1, 2, 5, 6, 7, 10, 11, 12, 13]]
regresyon_gds = sm.OLS(endog=y, exog=X_gds).fit()
regresyon_gds.summary()
Xr = train_dumm.loc[:,["TotalBsmtSF", "1stFlrSF", "GrLivArea","ExterQual_TA", "Foundation_PConc",
                       "BsmtQual_TA", "KitchenQual_TA", "GarageFinish_Unf",
                       "GarageCars_3.0"]].values

yr = train_corr.loc[:, "SalePrice"].values

Xr_test = X_test.loc[:,["TotalBsmtSF", "1stFlrSF", "GrLivArea","ExterQual_TA", "Foundation_PConc",
                       "BsmtQual_TA", "KitchenQual_TA", "GarageFinish_Unf", "GarageCars_3.0"]]
makine2 = LinearRegression(normalize=True)

makine2.fit(Xr, yr)

yr_pred = makine2.predict(Xr)

print("Root mean square error train = " + str(np.sqrt(mean_squared_error(yr, yr_pred))))
print("R2 score = " + str(r2_score(yr, yr_pred)))
yr_test = makine2.predict(Xr_test)
submission = pd.DataFrame({"Id": test_id})

submission["SalePrice"] = yr_test
submission.to_csv("submission.csv", index=False)