# This Python 3 environment comes with many helpful analytics libraries installed
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
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, r2_score
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
Submission = df_test['Id']
df_train.describe().T
#Get the data types of each variable
df_types = df_train.dtypes
corr = df_train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)#
print(corr.SalePrice)

ax = plt.subplots(ncols=1, figsize=(10,10))
corr_matrix = df_train.corr()
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, mask=mask, vmin = -1, vmax = 1, center = 0);
plt.show()
#Deep dive into one of our correlation pairs
sns.scatterplot(x= df_train['1stFlrSF'], y=df_train['TotalBsmtSF'])
#Distribution plot of SalePrice
sns.distplot(df_train['SalePrice'], color="b");
plt.show()

#Log transformed distribution plot of SalePrice
Log_Y = df_train['SalePrice']
sns.distplot(np.log10(Log_Y), color="c");
plt.show()
"""for i in df_train.select_dtypes(include='object').columns:
    sns.boxplot(x=df_train[i], y = df_train['SalePrice'])
    plt.xticks(rotation=90)
    #plt.show()
    
for i in df_train.select_dtypes(exclude='object').columns:
    sns.scatterplot(x=df_train[i], y=df_train['SalePrice'])
    #plt.show()"""
#Finding all the outliers and their corresponding rows
print(df_train[(df_train['LotFrontage']>300)].index)
print(df_train[(df_train['LotArea']>200000)].index)
print(df_train[(df_train['BsmtFinSF1']>5000)].index)
print(df_train[(df_train['TotalBsmtSF']>5000)].index)
print(df_train[(df_train['1stFlrSF']>4000)].index)
print(df_train[(df_train['GrLivArea']>4500)].index)
print(df_train[(df_train['EnclosedPorch']>500)].index)
print(df_train[(df_train['MiscVal']>8000)].index)
#I noticed that not dropping 934, 313 and 346 improved the RMSE score, but decided to exclude them anyway.
df_train = df_train.drop([523, 1298, 934, 313, 346, 197])
#Create new columns to identify the train and test dataset when it will be requried to split again
df_train['Train']=1
df_test['Train']=0

#Join datasets
df = pd.concat([df_train, df_test], axis=0)
#Change type of variables
df['MSSubClass'] = df['MSSubClass'].apply(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
df['YearRemodAdd'] = df['YearRemodAdd'].astype(str)
df = df.drop(["Id"], axis=1)
quantitative = [f for f in df.columns if df.dtypes[f] != 'object']
qualitative = [f for f in df.columns if df.dtypes[f] == 'object']

#Fill missing values for quantitative variables
for i in quantitative:
    df.fillna(df.median(), inplace = True)
    #print(i, df[i].median())
    
#Fill missing values for special variables
spec_categ_col =['PoolArea', 'Fence', 'MiscFeature', 'Alley','FireplaceQu']
for i in spec_categ_col:
    df[i] = df[i].fillna('None')
    
#Fill missing values for categorical variables
for i in qualitative:
    df[i].fillna(df[i].mode()[0], inplace = True)
    
#Check missing values for all variables
df.isnull().sum().sum()
df.skew(axis = 0, skipna = True).sort_values(ascending=False)
#Log Transform variables to make them better fitted to the regression model
Num = 1
if Num <= 1:
    df["LotArea"] = np.log1p(df["LotArea"])
    df["LotFrontage"] = np.log1p(df["LotFrontage"])
    df["GrLivArea"] = np.log1p(df["GrLivArea"])
else:
    num_feats = df.dtypes[df.dtypes != "object"].index
    skewed_feats =df[num_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    skewed_feats
    df[skewed_feats] = np.log1p(df[skewed_feats])
#Feature Engineering
#Total Floor area of entire house
df['TotalSF']=df['TotalBsmtSF']+ df['1stFlrSF']+ df['2ndFlrSF']
#Total number of baths
df['TotalBath'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
#Total porch area
df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch']+ df['WoodDeckSF']
#Create dummy variables for categorical columns.
df = pd.get_dummies(data=df)
#Split into training and test dataset from the original data
df_train = df[df["Train"] ==1]
df_test = df[df["Train"] ==0]
#Drop the unwanted columns
df_train = df_train.drop(["Train"], axis=1)
df_test = df_test.drop(["Train"], axis=1)
df_test = df_test.drop(["SalePrice"], axis=1)
if Num <= 1:
    y = np.log1p(df_train["SalePrice"]).values
else:
    y = df_train.SalePrice

x = df_train.drop(["SalePrice"], axis=1)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
scaler = MinMaxScaler()
scaler.fit(x)

scaled_x_train = scaler.transform(x)
scaled_x_test = scaler.transform(df_test)

x = pd.DataFrame(scaled_x_train, columns = x.columns)
df_test = pd.DataFrame(scaled_x_test, index = df_test.index, columns = df_test.columns)
Lasso_model = LassoCV(alphas = [1, 0.1,0.05, 0.001, 0.0005], selection='random', max_iter=15000).fit(x, y)
Lasso_train = Lasso_model.predict(x)
Lasso_Test = Lasso_model.predict(df_test)

print(np.sqrt(mean_squared_error(y,Lasso_train)))
print(r2_score(y, Lasso_train))
#print(np.sqrt(mean_squared_error(Validate,Lasso_Test)))
coef = pd.Series(Lasso_model.coef_, index = x.columns)
imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
imp_coef.plot(kind="barh")
plt.show()
#XGB Boosting Regressor
XGB_model = XGBRegressor( n_estimators=3000, 
                          learning_rate=0.01,
                          max_depth=10, 
                          max_features='sqrt',
                          min_samples_leaf=15, 
                          criterion='mse', 
                          min_samples_split=10, 
                          random_state =10).fit(x,y)

XGB_train = gbr_model.predict(x)
XGB_Test = gbr_model.predict(df_test)

print(np.sqrt(mean_squared_error(y,XGB_train)))
print(r2_score(y, XGB_train))
#print(np.sqrt(mean_squared_error(Validate,XGB_Test)))
#Random Forest Regressor
forest_model = RandomForestRegressor(
    random_state=10, 
    n_estimators=3000,
    max_depth=10, 
    max_features='sqrt',
    min_samples_leaf=15, 
    criterion='mse', 
    min_samples_split=10)

forest_model.fit(x,y)
forest_train = forest_model.predict(x)
Forest_Test = forest_model.predict(df_test)

print(np.sqrt(mean_squared_error(y,forest_train)))
print(r2_score(y, forest_train))
#print(np.sqrt(mean_squared_error(Validate,Forest_Test)))
#Random Forest Regressor2

forest_model2 = RandomForestRegressor(random_state=1)
forest_model2.fit(x,y)
forest_train2 = forest_model.predict(x)
Forest_Test2 = forest_model.predict(df_test)
print(np.sqrt(mean_squared_error(y,forest_train2)))
print(r2_score(y, forest_train2))
#Gradient Boosting Regressor
gbr_model = ensemble.GradientBoostingRegressor(
    n_estimators=3000, 
    learning_rate=0.01,
    max_depth=10, 
    max_features='sqrt',
    min_samples_leaf=15, 
    criterion='mse', 
    min_samples_split=10, 
    random_state =10).fit(x,y)

gbr_train = gbr_model.predict(x)
gbr_Test = gbr_model.predict(df_test)

print(np.sqrt(mean_squared_error(y,gbr_train)))
print(r2_score(y, gbr_train))
#print(np.sqrt(mean_squared_error(Validate,gbr_Test)))
#Ridge Regression
Ridge_model = Ridge(alpha=0.05)
Ridge_model.fit(x, y)
Ridge_train = Ridge_model.predict(x)
Ridge_Test = Ridge_model.predict(df_test)

#print(pd.Series(Ridge_model.coef_, index = x.columns))

print(np.sqrt(mean_squared_error(y,Ridge_train)))
print(r2_score(y, Ridge_train))
#print(np.sqrt(mean_squared_error(Validate,Ridge_Test)))
#Gradient Boosting Regressor
gbr_model = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01,
                                   max_depth=10, max_features='sqrt',
                                   min_samples_leaf=15, criterion='mse', min_samples_split=10, random_state =10).fit(x,y)
gbr_train = gbr_model.predict(x)
gbr_Test = gbr_model.predict(df_test)

#print(np.sqrt(mean_squared_error(Validate,gbr_Test)))
print(np.sqrt(mean_squared_error(y,gbr_train)))
print(r2_score(y, gbr_train))
#Ridge Regression
Ridge_model = Ridge(alpha=0.05)
Ridge_model.fit(x, y)
Ridge_train = Ridge_model.predict(x)
Ridge_Test = Ridge_model.predict(df_test)

#print(pd.Series(Ridge_model.coef_, index = x.columns))

print(np.sqrt(mean_squared_error(y,Ridge_train)))
print(r2_score(y, Ridge_train))
#print(np.sqrt(mean_squared_error(Validate,Ridge_Test)))
Hybrid_train = 0.4*Lasso_train + 0.2*gbr_train + 0.4*XGB_train
print(np.sqrt(mean_squared_error(y,Hybrid_train)))
print(r2_score(y, Hybrid_train))
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(y,Ridge_train, label ='Ridge')
ax1.scatter(y,gbr_train, label ='GBR')
ax1.scatter(y,Lasso_train, label ='Lasso')
ax1.scatter(y,forest_train, label ='RnDForest')
ax1.scatter(y,XGB_train, label ='XGB')
ax1.scatter(y,Hybrid_train, label ='Hybrid')
plt.legend(loc='upper left')
plt.plot([10.5, 13.5], [10.5, 13.5], c = "black")
plt.show()
#Residuals
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(y,Ridge_train-y, label ='Ridge')
ax1.scatter(y,gbr_train-y, label ='GBR')
ax1.scatter(y,Lasso_train-y, label ='Lasso')
ax1.scatter(y,forest_train2-y, label ='RnDForest')
ax1.scatter(y,XGB_train-y, label ='XGB')
ax1.scatter(y,Hybrid_train-y, label = 'Hybrid')
plt.legend(loc='upper left')
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "black")
plt.show()
df = pd.DataFrame({'Predicted': Lasso_Test})
df = np.exp(df)
Sub = pd.DataFrame()
Sub['Id'] = Submission
Sub['SalePrice'] = df
Sub.to_csv('Input URL Here!')

