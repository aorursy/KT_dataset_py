
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


hp=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
hp_test= pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
hp.info()
hp.head()
hp.describe()
fig = plt.figure(figsize = (10,5))
sns.distplot(hp['SalePrice'])
def converter(garage):
    if garage==0:
        return 0
    else:
        return 1
hp['GarageYN'] = hp['GarageArea'].apply(converter)
hp.head()
sns.scatterplot(data=hp, x="GrLivArea", y="SalePrice", hue="GarageYN")
sns.set_style('darkgrid')
g = sns.FacetGrid(hp,hue="GarageYN",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'GrLivArea',bins=20,alpha=0.7)
var = 'OverallQual'
data = pd.concat([hp['SalePrice'], hp[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
fig = plt.figure(figsize = (25,15))
sns.heatmap(hp.corr(), annot =True, fmt='.1f')
mostcorrelated= hp[["OverallQual","GrLivArea", "FullBath", "TotalBsmtSF", "1stFlrSF", "GarageCars", "GarageArea","YearBuilt", "YearRemodAdd", "MasVnrArea","TotRmsAbvGrd",

"Fireplaces", "GarageYrBlt", "SalePrice"]]
fig = plt.figure(figsize = (15,15))
sns.heatmap(mostcorrelated.corr(), annot =True, fmt='.1f')
hpmodel=hp.drop(["TotRmsAbvGrd", "GarageYrBlt", "1stFlrSF", "GarageCars"], axis=1)
total = hp.isnull().sum().sort_values(ascending=False)
percent = (hp.isnull().sum()/hp.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(18)
hp['PoolQC'].fillna('No Pool', inplace=True)
hp['MiscFeature'].fillna('No Other Features', inplace=True)
hp['Alley'].fillna('No Alley Access', inplace =True)
hp['Fence'].fillna('No Fence', inplace =True)
hp['FireplaceQu'].fillna('No Fireplace', inplace =True)
hp['LotFrontage'].fillna(hp['LotFrontage'].mean(), inplace =True)
hp['GarageQual'].fillna('No Garage', inplace =True)
hp['GarageCond'].fillna('No Garage', inplace =True)
hp['GarageType'].fillna('No Garage', inplace =True)
hp['GarageFinish'].fillna('No Garage', inplace =True)
hp['BsmtFinType2'].fillna('No Basement', inplace =True)
hp['BsmtCond'].fillna('No Basement', inplace =True)
hp['BsmtExposure'].fillna('No Basement', inplace =True)
hp['BsmtQual'].fillna('No Basement', inplace =True)
hp['BsmtFinType1'].fillna('No Basement', inplace =True)
hp['MasVnrArea'].fillna(0, inplace =True)
hp['MasVnrType'].fillna('No Masonary', inplace =True)
hp['Electrical'].fillna('SBrkr', inplace =True)
hp['Electrical'].value_counts()
total = hpmodel.isnull().sum().sort_values(ascending=False)
percent = (hpmodel.isnull().sum()/hp.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(18)
hpmodel['GrLivArea'].sort_values(ascending=False)
hpmodel = hpmodel .drop(hpmodel[hpmodel ['Id'] == 1299].index)
hpmodel  = hpmodel .drop(hpmodel [hpmodel['Id'] == 524].index)
sns.distplot(hpmodel['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(hpmodel['SalePrice'], plot=plt)
hpdeneme=hpmodel
hpdeneme.select_dtypes("object").columns
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
result = encoder.fit_transform(hpdeneme[['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']])
print(result)
hpdeneme[['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition']]=result
hpdeneme.head()
Y =hpdeneme["SalePrice"]
X =hpdeneme.drop(["SalePrice", "Id"],axis =1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
lr.coef_
coeff_df = pd.DataFrame(lr.coef_,x_train.columns,columns=['Coefficient'])
coeff_df
y_pred = lr.predict(x_test)
lr.intercept_
lr.score(x_test,y_test)
from sklearn import metrics
from math import sqrt
print('MAE: {}'.format(metrics.mean_absolute_error(y_test, y_pred)))
print('MSE: {}'.format(metrics.mean_squared_error(y_test, y_pred)))
print('RMSE: {}'.format(sqrt(metrics.mean_squared_error(y_test, y_pred))))
print("R2: {}".format(metrics.r2_score(y_test,y_pred)))
myprediction=lr.predict(hp_test)
hp_test.info()
my_submission = pd.DataFrame({'Id': hp_test.Id, 'SalePrice':hp_test.SaleP})
#my_submission.to_csv('submission.csv', index=False)