#importing essential libraries.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization
#from fancyimpute import KNN #KNN imputation
import fancyimpute
from sklearn.cluster import DBSCAN  #outlier detection
from collections import Counter
import matplotlib# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dfte = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_test.head()
dfte.head()
del dfte['Id']
del df_test['Id']
dfte.head()
dfte.info()
dfte.groupby('LandContour')['SalePrice'].mean().reset_index()
dfte.describe()
dfte.isnull().sum()
#Sample Impute by mode in the column Lot Area
mode = dfte.LotFrontage.mode().iloc[0]
df_mode = dfte.LotFrontage.fillna(mode)
fig, axes = plt.subplots(nrows=1,ncols=2)
dfte.LotFrontage.hist(bins = 30, ax = axes[0])
df_mode.hist(bins = 30, ax = axes[1], color = 'red')
print(dfte.LotFrontage.describe())
print(df_mode.describe())
#Sample Impute by mean in the column Lot Area
mean = dfte.LotFrontage.mean()
df_mean = dfte.LotFrontage.fillna(mean)
fig, axes = plt.subplots(nrows=1,ncols=2)
dfte.LotFrontage.hist(bins = 30, ax = axes[0])
df_mean.hist(bins = 30, ax = axes[1], color = 'red')
print(dfte.LotFrontage.describe())
print(df_mean.describe())
#Sample Impute by median in the column Lot Area
median = dfte.LotFrontage.median()
df_median = dfte.LotFrontage.fillna(median)
fig, axes = plt.subplots(nrows=1,ncols=2)
dfte.LotFrontage.hist(bins = 30, ax = axes[0])
df_median.hist(bins = 30, ax = axes[1], color = 'red')
print(dfte.LotFrontage.describe())
print(df_median.describe())
#Sample Impute by knn in the column Lot Area
from fancyimpute import KNN
dfte.LotFrontage = dfte.LotFrontage.astype('float')
dfnumeric = dfte.select_dtypes('float')
df_knn = KNN(k=5).complete(dfnumeric)
df_knn = pd.DataFrame(df_knn)
df_knn.shape
df_test.LotFrontage = df_test.LotFrontage.astype('float')
dfnumeric2 = df_test.select_dtypes('float')
df_knn2 = KNN(k=5).complete(dfnumeric2)
df_knn2 = pd.DataFrame(df_knn2)
df_knn2.shape
df_knn.index = dfnumeric.index
df_knn.columns = dfnumeric.columns
print(dfte.LotFrontage.describe())
print(df_knn.LotFrontage.describe())
df_knn2.index = dfnumeric2.index
df_knn2.columns = dfnumeric2.columns
print(df_test.LotFrontage.describe())
print(df_knn2.LotFrontage.describe())
dfim = dfte.copy()
dfim.LotFrontage = df_knn.LotFrontage
dfim.Alley = dfte.Alley.fillna(mode)
dfim.MasVnrType = dfte.MasVnrType.fillna(mode)
dfim.MasVnrArea = df_knn.MasVnrArea
dfim.FireplaceQu = dfte.FireplaceQu.fillna(mode)
dfim.GarageType = dfte.GarageType.fillna(mode)
dfim.GarageYrBlt = dfte.GarageYrBlt.fillna(mode)
dfim.GarageFinish = dfte.GarageFinish.fillna(mode)
dfim.GarageQual = dfte.GarageQual.fillna(mode)
dfim.GarageCond = dfte.GarageCond.fillna(mode)
dfim.PoolQC = dfte.PoolQC.fillna(mode)
dfim.Fence = dfte.Fence.fillna(mode)
dfim.MiscFeature = dfte.MiscFeature.fillna(mode)
dfim.head()
dfimte = df_test.copy()
dfimte.LotFrontage = df_knn2.LotFrontage
dfimte.Alley = df_test.Alley.fillna(mode)
dfimte.MasVnrType = df_test.MasVnrType.fillna(mode)
dfimte.MasVnrArea = df_knn2.MasVnrArea
dfimte.FireplaceQu = df_test.FireplaceQu.fillna(mode)
dfimte.GarageType = df_test.GarageType.fillna(mode)
dfimte.GarageYrBlt = df_test.GarageYrBlt.fillna(mode)
dfimte.GarageFinish = df_test.GarageFinish.fillna(mode)
dfimte.GarageQual = df_test.GarageQual.fillna(mode)
dfimte.GarageCond = df_test.GarageCond.fillna(mode)
dfimte.PoolQC = df_test.PoolQC.fillna(mode)
dfimte.Fence = df_test.Fence.fillna(mode)
dfimte.MiscFeature = df_test.MiscFeature.fillna(mode)
dfimte.head()
dfim_labeled = dfim.copy()
cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
dfim_labeled = pd.get_dummies(dfim_labeled, columns = cols, drop_first = True)
dfim_labeled.head()
dfim_labte = dfimte.copy()
dfim_labte = pd.get_dummies(dfim_labte, columns = cols, drop_first = True)
dfim_labte.head()
#Outlier Treatment on train dataset by DBSCAN
ous = ['LotArea', 'MasVnrArea', 'GarageArea']
data1 = dfim_labeled[ous]
model_train = DBSCAN(eps = 1000, min_samples = 19).fit(data1)
outliers_df = pd.DataFrame(data1)
print(Counter(model_train.labels_))
df_train_out = dfim_labeled.copy()
df_train_out = df_train_out[model_train.labels_ == 0]
df_test_out = dfim_labte.copy()
df_test_out['SalePrice'] = df_train_out['SalePrice'].mean()
df_test_out.head()
X = df_train_out.iloc[:, df_train_out.columns != 'SalePrice']
X.head()
Y = df_train_out.iloc[:, df_train_out.columns == 'SalePrice']
Y.head()
#Splitting training dataset into sample train and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
#Fitting Multiple Linear Regression 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
rms = sqrt(mean_squared_error(Y_test, y_pred))
rms
y_resid = y_pred - Y_test
plt.plot(Y_test, y_resid,  "o")
plt.xlabel('fitted')
plt.ylabel('residuals')
plt.show()
X_train.shape
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
regressor_poly = LinearRegression()
regressor_poly.fit(X_poly, Y_train)
y_pred_poly = regressor.predict(X_test)
rms = sqrt(mean_squared_error(Y_test, y_pred_poly))
rms
y_resid = y_pred_poly - Y_test
plt.plot(Y_test, y_resid,  "o")
plt.xlabel('fitted')
plt.ylabel('residuals')
plt.show()
#Cross validating dataset using K-Fold CV 
import nltk
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score
kf = KFold(n_splits=5)
sum = 0
X = df_train_out.iloc[:, df_train_out.columns != 'SalePrice']
Y = df_train_out.iloc[:, df_train_out.columns == 'SalePrice']
reg = LinearRegression()
X_poly = poly_reg.fit_transform(X)
scores = cross_val_score(reg, X_poly, Y, cv = 10, scoring = 'mean_squared_error')
print(scores)

mse_scores = -scores
rmse_scores = np.sqrt(mse_scores)
print(rmse_scores.mean())
X = df_train_out.iloc[:, df_train_out.columns != 'SalePrice']
Y = df_train_out.iloc[:, df_train_out.columns == 'SalePrice']
X_testo = dfim_labte
reg.fit(X, Y)
pred_sale_price = reg.predict(X_testo)
pred_sale_price