#Useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, norm, boxcox_normmax
from scipy.special import boxcox1p
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sns.distplot(train['SalePrice'])
print('Skewness in SalePrice: {0}'.format(train['SalePrice'].skew()))
corr = train.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, square=True)
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', alpha=0.5, ylim=(0,800000));
train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'])
df = pd.concat([train, test], axis=0)
df.shape
# Non-numeric features stored as numericals. Converting to string.
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
df['MSSubClass'] = df['MSSubClass'].apply(str)
categorical_features = df.select_dtypes(include=['object'])
numerical_features =df.select_dtypes(exclude=['object'])
# Missing values in numerical features
numerical_features.isnull().sum().sort_values(ascending=False)[:15]

#1459 missing in SalePrice is because, test set don't have SalePrice column.
corr['SalePrice'].sort_values(ascending=False)
data = pd.concat([train['YearBuilt'], train['GarageYrBlt']], axis=1)
data.head(10)
numerical_features = numerical_features.drop(columns=['GarageYrBlt'], axis=1)
numerical_features['LotFrontage'] = numerical_features['LotFrontage'].fillna(68)
NA_columns = ['MasVnrArea','BsmtHalfBath', 'BsmtFullBath','BsmtFinSF1','TotalBsmtSF','BsmtUnfSF','BsmtFinSF2','GarageCars','GarageArea']
numerical_features[NA_columns]= numerical_features[NA_columns].fillna(0)
total = categorical_features.isnull().sum().sort_values(ascending=False)
percent = (categorical_features.isnull().sum()/categorical_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
categorical_features = categorical_features.drop(columns=['PoolQC','MiscFeature','Alley','Fence'])
obj_NA_columns = ['FireplaceQu','GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType']
categorical_features[obj_NA_columns]= categorical_features[obj_NA_columns].fillna('None')

columns_with_lowNA = ['MSZoning','Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType', 'Electrical']
for i in columns_with_lowNA:
  categorical_features[i] = categorical_features[i].fillna(categorical_features[i].mode()[0])
categorical_features = categorical_features.drop(['Heating','RoofMatl','Condition2','Street','Utilities'],axis=1)
binary_map  = {'TA':2,'Gd':3, 'Fa':1,'Ex':4,'Po':1,'None':0,'Y':1,'N':0,'Reg':3,'IR1':2,'IR2':1,'IR3':0,"None" : 0,
            "No" : 2, "Mn" : 2, "Av": 3,"Gd" : 4,"Unf" : 1, "LwQ": 2, "Rec" : 3,"BLQ" : 4, "ALQ" : 5, "GLQ" : 6}

categorical_features['ExterQual'] = categorical_features['ExterQual'].map(binary_map)
categorical_features['ExterCond'] = categorical_features['ExterCond'].map(binary_map)
categorical_features['BsmtCond'] = categorical_features['BsmtCond'].map(binary_map)
categorical_features['BsmtQual'] = categorical_features['BsmtQual'].map(binary_map)
categorical_features['HeatingQC'] = categorical_features['HeatingQC'].map(binary_map)
categorical_features['KitchenQual'] = categorical_features['KitchenQual'].map(binary_map)
categorical_features['FireplaceQu'] = categorical_features['FireplaceQu'].map(binary_map)
categorical_features['GarageQual'] = categorical_features['GarageQual'].map(binary_map)
categorical_features['GarageCond'] = categorical_features['GarageCond'].map(binary_map)
categorical_features['CentralAir'] = categorical_features['CentralAir'].map(binary_map)
categorical_features['LotShape'] = categorical_features['LotShape'].map(binary_map)
categorical_features['BsmtExposure'] = categorical_features['BsmtExposure'].map(binary_map)
categorical_features['BsmtFinType1'] = categorical_features['BsmtFinType1'].map(binary_map)
categorical_features['BsmtFinType2'] = categorical_features['BsmtFinType2'].map(binary_map)

PDrive =   {"N" : 0, "P" : 1, "Y" : 2}
categorical_features['PavedDrive'] = categorical_features['PavedDrive'].map(PDrive)

rest_categorical_features = categorical_features.select_dtypes(include=['object'])
#Using One hot encoder
categorical_features = pd.get_dummies(categorical_features, columns=rest_categorical_features.columns) 
df = pd.concat([numerical_features, categorical_features], axis=1)
skewed_features = []
for i in numerical_features:
  skewed_features.append([i, numerical_features[i].skew()])
# threshold skewness of 0.5
high_skewed = []
for i in range(len(skewed_features)):
  if skewed_features[i][1] > 0.5:
    high_skewed.append(skewed_features[i])
high_skewed = ['LotArea', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF',
 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

#transform skewed features
for i in high_skewed:
  df[i] = boxcox1p(df[i], boxcox_normmax(df[i]+1))
df_train = df.iloc[:1458, :]
df_test = df.iloc[1458:, :]
df_test.drop(columns='SalePrice', axis=1, inplace=True)
print(df_train.shape, df_test.shape)
targets = df_train['SalePrice']
df_final = df_train.drop(columns='SalePrice', axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_final, targets, test_size= 0.2, random_state=0)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f'Predicted: {y_pred[:5]}')
print(f'true: {list(y_test[:5])}')
print('\n')

mse = mean_squared_error(y_test, y_pred)
lr_rmse = np.sqrt(mse)
print(f'Root mean squared error: {lr_rmse}')
print('\n')

print(f'Training score: {model.score(x_train, y_train)}')
print(f'Testing score: {model.score(x_test, y_test)}')
from lightgbm import LGBMRegressor
model = LGBMRegressor()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f'Predicted: {y_pred[:5]}')
print(f'true: {list(y_test[:5])}')
print('\n')

mse = mean_squared_error(y_test, y_pred)
lgbm_rmse = np.sqrt(mse)
print(f'Root mean squared error: {lgbm_rmse}')
print('\n')

print(f'Training score: {model.score(x_train, y_train)}')
print(f'Testing score: {model.score(x_test, y_test)}')
from sklearn.linear_model import Ridge
model = Ridge(alpha=1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(f'Predicted: {y_pred[:5]}')
print(f'true: {list(y_test[:5])}')
print('\n')

mse = mean_squared_error(y_test, y_pred)
rid_rmse = np.sqrt(mse)
print(f'Root mean squared error: {rid_rmse}')
print('\n')

print(f'Training score: {model.score(x_train, y_train)}')
print(f'Testing score: {model.score(x_test, y_test)}')
print('Root Mean Squared Error')
print('=======================')
print(f'Ridge: {rid_rmse} (best)')
print(f'LinearRegressor: {lr_rmse}')
print(f'LGBMRegressor: {lgbm_rmse}')
final_pred = model.predict(df_test)
final_pred = final_pred*10000
final_pred
predictions=pd.DataFrame(final_pred)
sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sub = pd.concat([sample['Id'],predictions], axis=1)
sub.columns=['Id','SalePrice']
sub.to_csv('My_submission2.csv',index=False)