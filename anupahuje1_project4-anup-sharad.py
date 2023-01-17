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
!pip install regressors
import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

import statsmodels.formula.api as sm

import statsmodels.sandbox.tools.cross_val as cross_val

from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model as lm

from regressors import stats

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut



print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv')

df.columns
cols = ['SalePrice','Neighborhood', 'OverallQual','GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt','HouseStyle','TotRmsAbvGrd']

df_train = df[cols]

df_train.head()
sns.pairplot(df_train)
df_train.corr()
print("Check for NaN/null values:\n",df_train.isnull().values.any())

print("Number of NaN/null values:\n",df_train.isnull().sum())
main = sm.ols(formula="SalePrice ~ (OverallQual*GrLivArea)+(YearBuilt*OverallQual)+(Neighborhood*HouseStyle*TotalBsmtSF)+(TotalBsmtSF*TotRmsAbvGrd)+(FullBath*TotRmsAbvGrd*GarageArea)",data=df_train).fit()

print(main.summary())
main = sm.ols(formula="SalePrice ~ (OverallQual*GrLivArea)+(Neighborhood*HouseStyle*TotalBsmtSF)+(TotalBsmtSF*TotRmsAbvGrd*GarageArea)",data=df_train).fit()

print(main.summary())
main = sm.ols(formula="SalePrice ~ OverallQual+I(OverallQual*OverallQual)+I(OverallQual*OverallQual*OverallQual)",data=df_train).fit()

print(main.summary())
main = sm.ols(formula="SalePrice ~ GrLivArea+I(GrLivArea*GrLivArea)+I(GrLivArea*GrLivArea*GrLivArea)",data=df_train).fit()

print(main.summary())
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
df_dum = pd.get_dummies(df_train)

df_dum.head()
df_dum.columns
inputDF = df_dum[['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF',

       'FullBath', 'YearBuilt', 'TotRmsAbvGrd', 'Neighborhood_Blmngtn',

       'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide',

       'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor',

       'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR',

       'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes',

       'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge',

       'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU',

       'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst',

       'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker',

       'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_1Story',

       'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story',

       'HouseStyle_SFoyer', 'HouseStyle_SLvl']]

outputDF = df_dum[["SalePrice"]]



model_fwd = sfs(LinearRegression(),k_features=8,forward=True,verbose=2,cv=8,n_jobs=-1,scoring='r2')

model_fwd.fit(inputDF,outputDF)
model_fwd.k_feature_names_
model_bkd = sfs(LinearRegression(),k_features=8,forward=False,verbose=2,cv=8,n_jobs=-1,scoring='r2')

model_bkd.fit(inputDF,outputDF)
model_bkd.k_feature_names_
from sklearn import metrics

from sklearn.linear_model import LinearRegression
inputDF_looc = df_dum[['OverallQual',

 'GrLivArea',

 'GarageArea',

 'YearBuilt',

 'Neighborhood_NoRidge',

 'Neighborhood_NridgHt',

 'Neighborhood_StoneBr',

 'HouseStyle_1Story']]

outputDF_looc = df_dum[["SalePrice"]]

model_fwd_sfs = LinearRegression()

loocv = LeaveOneOut()



rmse = np.sqrt(-cross_val_score(model_fwd_sfs, inputDF_looc, outputDF_looc, scoring="neg_mean_squared_error", cv = loocv))

print(rmse.mean())
kf = KFold(5, shuffle=True, random_state=42).get_n_splits(inputDF_looc)

rmse = np.sqrt(-cross_val_score(model_fwd_sfs, inputDF_looc, outputDF_looc, scoring="neg_mean_squared_error", cv = kf))

print(rmse.mean())
kf = KFold(10, shuffle=True, random_state=42).get_n_splits(inputDF_looc)

rmse = np.sqrt(-cross_val_score(model_fwd_sfs, inputDF_looc, outputDF_looc, scoring="neg_mean_squared_error", cv = kf))

print(rmse.mean())
inputDF_looc_bkd = df_dum[['OverallQual',

 'GrLivArea',

 'GarageArea',

 'Neighborhood_NAmes',

 'Neighborhood_NWAmes',

 'Neighborhood_OldTown',

 'Neighborhood_SWISU',

 'HouseStyle_1Story']]

outputDF_looc_bkd = df_dum[["SalePrice"]]

model_bkd_sfs = LinearRegression()

loocv = LeaveOneOut()



rmse = np.sqrt(-cross_val_score(model_bkd_sfs, inputDF_looc_bkd, outputDF_looc_bkd, scoring="neg_mean_squared_error", cv = loocv))

print(rmse.mean())
kf = KFold(5, shuffle=True, random_state=42).get_n_splits(inputDF_looc_bkd)

rmse = np.sqrt(-cross_val_score(model_bkd_sfs, inputDF_looc_bkd, outputDF_looc_bkd, scoring="neg_mean_squared_error", cv = kf))

print(rmse.mean())
kf = KFold(10, shuffle=True, random_state=42).get_n_splits(inputDF_looc_bkd)

rmse = np.sqrt(-cross_val_score(model_bkd_sfs, inputDF_looc_bkd, outputDF_looc_bkd, scoring="neg_mean_squared_error", cv = kf))

print(rmse.mean())
inputDF_final = df_dum[['OverallQual',

 'GrLivArea',

 'GarageArea',

 'YearBuilt',

 'Neighborhood_NoRidge',

 'Neighborhood_NridgHt',

 'Neighborhood_StoneBr',

 'HouseStyle_1Story']]

outputDF_final = df_dum[["SalePrice"]]

model_final = LinearRegression()

loocv = LeaveOneOut()

rmse = np.sqrt(-cross_val_score(model_final, inputDF_final, outputDF_final, scoring="neg_mean_squared_error", cv = loocv))

print(rmse.mean())

model_lm = lm.LinearRegression()

results = model_lm.fit(inputDF_final,outputDF_final)



print("P-value:\n",stats.coef_pval(model_lm, inputDF_final, outputDF_final))

print("Adjusted R-Squared:\n",stats.adj_r2_score(model_lm, inputDF_final, outputDF_final))
main = sm.ols(formula="SalePrice ~ I(OverallQual*OverallQual*OverallQual)+I(GrLivArea*GrLivArea)+(OverallQual*GrLivArea)+(Neighborhood*HouseStyle*TotalBsmtSF)+(TotalBsmtSF*TotRmsAbvGrd*GarageArea)",data=df_train).fit()

print(main.summary())