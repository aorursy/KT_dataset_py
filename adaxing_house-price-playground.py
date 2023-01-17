# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import re

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.options.display.max_columns = None
pd.options.display.max_rows = 80

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
def inv_y(trans_y):
    return np.exp(trans_y)
def check_null(df, n_head=30):
    return df.isna().sum().sort_values(ascending=False).head(n_head)
def score(model, train_x, train_y, val_x, val_y):
    score = {}
    model.fit(train_x, train_y)
    preds = model.predict(val_x)
    score['mae'] = mean_absolute_error(inv_y(preds), inv_y(val_y))
    score['mse'] = mean_squared_error(inv_y(preds), inv_y(val_y))
    return score
train
test.head()
train.info()
train.describe()
train['GrLivArea'].describe()
train.describe(include='O')
pd.DataFrame(train.isnull().sum().sort_values(ascending=False)).reset_index()[:20]
plt.figure(figsize=(10, 5))
plt.xticks(np.arange(min(train.YearBuilt.tolist()), max(train.YearBuilt.tolist())+1, 7.0))
plt.bar(train.YearBuilt, train.SalePrice)
# 1. older buiding lower house price -> YearBuilt vs. SalePrice ?
train[['YearBuilt', 'SalePrice']].groupby(['YearBuilt']).mean().sort_values(by='SalePrice', ascending=False)
plt.bar(train.SaleType, train.SalePrice)
train[['SaleType', 'SalePrice']].groupby(['SaleType'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
# 2. SaleType vs. SalePrice relation with interest or new may relatively higher price?
plt.bar(train.SaleCondition, train.SalePrice)
train[['SaleCondition', 'SalePrice']].groupby(['SaleCondition'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
train['ExtraFeature'] = train.MiscFeature.notnull().astype('int')
plt.bar(train.ExtraFeature, train.SalePrice)
train[['ExtraFeature', 'SalePrice']].groupby(['ExtraFeature'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
# 3. have value in MiscFeature may increase house price as it contain several services?
# no, nothing extra is twice of having extra in price
# can drop MiscFeature columns, but need to consider as 1406/1460,almost 96% do not have values
train[train.PoolArea!=0]
# 4. pool also another indicator of higher price
# 1453/1460 do not have poolarea, 7 has, also not high price, not good indicator
# can drop PoolArea, PoolQC columns
# 5. OverallCond/OverallQual feature store as numerical feature, should be converted to onehot encoder 
plt.bar(train.OverallCond, train.SalePrice)
train[['OverallCond', 'SalePrice']].groupby(['OverallCond'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
plt.bar(train.OverallQual, train.SalePrice)
train[['OverallQual', 'SalePrice']].groupby(['OverallQual'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
# overallQuality as good indicator, overallCond may be dropped
plt.figure(figsize=(23, 5))
plt.bar(train.Neighborhood, train.SalePrice)
train[['Neighborhood', 'SalePrice']].groupby(['Neighborhood'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
# 6. Neighbohood could be good indicator of house price
# can use label encoder as there is ordinal 
plt.bar(train.LandSlope, train.SalePrice)
train[['LandSlope', 'SalePrice']].groupby(['LandSlope'], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
# 7. LandSlope involve slope may lead expense of others increase, overall houseprice increase?
# Gtl has highest value&Sev has lowest value,  but overall Sev has higher value
corr = train.corr()
plt.subplots(figsize=(14, 12))
sns.heatmap(corr)
corr['SalePrice'].sort_values(ascending=False).head(15)
corr['SalePrice']['Fireplaces']
corr[corr.mask(np.eye(len(corr), dtype=bool)).abs()>0.5].SalePrice.sort_values(ascending=False)
# exploring 
eda_potential_features = ['YearBuilt', 'SaleType', 'OverallQual', 'Neighborhood', 'LandSlope']
eda_drop_features = ['SaleCondition', 'MiscFeature', 'ExtraFeature', 'PoolArea', 'OverallCond']
# potential features according correlation 
corr_potential_features = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF',
                      '1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']
a = 'Street Alley LotShape LandContour Utilities LotConfig LandSlope Neighborhood Condition1 Condition2 BldgType HouseStyle RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType,ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure BsmtFinType1 Heating HeatingQC CentralAir Electrical,Functional,GarageQual GarageCond PavedDrive, PoolQC Fence MiscFeature, YrSold SaleType SaleCondition'
remain_feature_list = [i for i in re.split(' |,', a) if len(i)]
remain_feature_list
def evaluation_model(model, train_x, train_y, valid_x, valid_y):
    score = {}
    model.fit(train_x, train_y)
    valid_prediction = model.predict(valid_x)
    score['mae'] = mean_absolute_error(valid_y, valid_prediction)
    score['mse'] = mean_squared_error(valid_y, valid_prediction)
    score['r2'] = r2_score(valid_y, valid_prediction)
    return score
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
y = train.SalePrice
X = train.loc[:, train.columns!='SalePrice']
x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=1)
x_train = x_train.select_dtypes(exclude='O')
x_valid = x_valid.select_dtypes(exclude='O')
drop_col = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
x_train = x_train.drop(drop_col, axis=1)
x_valid = x_valid.drop(drop_col, axis=1)
# baseline with DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=1)
evaluation_model(model, x_train, y_train, x_valid, y_valid)
# baseline with RandomForestRegressor
model = RandomForestRegressor(random_state=1)
evaluation_model(model, x_train, y_train, x_valid, y_valid)
# n_estimators
n_est = [i for i in range(290, 330, 10)]
mae_scores= []
mse_scores = []
r2_scores = []
for i in n_est:
    model = RandomForestRegressor(n_estimators=i, random_state=1)
    mae_scores.append(evaluation_model(model, x_train, y_train, x_valid, y_valid)['mae'])
    mse_scores.append(evaluation_model(model, x_train, y_train, x_valid, y_valid)['mse'])
    r2_scores.append(evaluation_model(model, x_train, y_train, x_valid, y_valid)['r2'])
    
# plt.plot(n_est, mae_scores, label='mae score')
# plt.plot(n_est, mse_scores, label='mse score')
plt.plot(n_est, r2_scores, label='r2 score')
model = RandomForestRegressor(n_estimators=310, random_state=1)
evaluation_model(model, x_train, y_train, x_valid, y_valid)
# depth
depth_ = [i for i in range(8, 17)]
mae_scores= []
mse_scores = []
r2_scores = []
for i in depth_:
    model = RandomForestRegressor(n_estimators=310, max_depth=i, random_state=1)
    mae_scores.append(evaluation_model(model, x_train, y_train, x_valid, y_valid)['mae'])
    mse_scores.append(evaluation_model(model, x_train, y_train, x_valid, y_valid)['mse'])
    r2_scores.append(evaluation_model(model, x_train, y_train, x_valid, y_valid)['r2'])
    
# plt.plot(depth_, mae_scores, label='mae score')
# plt.plot(depth_, mse_scores, label='mse score')
plt.plot(depth_, r2_scores, label='r2 score')
model = RandomForestRegressor(n_estimators=310, max_depth=12, random_state=1)
evaluation_model(model, x_train, y_train, x_valid, y_valid)
test.Id
x_train
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
y = train.SalePrice
X = train.loc[:, train.columns!='SalePrice']
X = X.select_dtypes(exclude='O')
drop_col = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
X = X.drop(drop_col, axis=1)
model_on_full_data = RandomForestRegressor(n_estimators=310, max_depth=12, random_state=1)
model_on_full_data = model.fit(X, y)
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
Id = test.Id
test = test.select_dtypes(exclude='O')
drop_col = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
test = test.drop(drop_col, axis=1)
test.fillna(0, inplace=True)
test_preds = model_on_full_data.predict(test)
output = pd.DataFrame({'Id': Id, 'SalePrice': test_preds})
output.to_csv('houseprice_test1.csv', index=False)
# another try with whole dataset k-fold
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
x_train = train.select_dtypes(exclude='O')
drop_col = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
x_train = x_train.drop(drop_col, axis=1)
model = RandomForestRegressor(n_estimators=310, max_depth=12, random_state=1)
score = {}
score['mae'] = np.mean(cross_val_score(model, x_train, train.SalePrice, cv=5, scoring="neg_mean_absolute_error"))
score['mse'] = np.mean(cross_val_score(model, x_train, train.SalePrice, cv=5, scoring="neg_mean_squared_error")) 
score['r2'] = np.mean(cross_val_score(model, x_train, train.SalePrice, cv=5))
score
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
corr_potential_features = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF',
                      '1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']
train[corr_potential_features].describe()
train[corr_potential_features].info()
train[corr_potential_features].isnull().any()
X = train[corr_potential_features]
y = train.SalePrice
x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=1)
# previous best model to see how's performance on new features
model = RandomForestRegressor(n_estimators=310, max_depth=12, random_state=1)
evaluation_model(model, x_train, y_train, x_valid, y_valid)
model = RandomForestRegressor(n_estimators=310, max_depth=12, random_state=1)
test_preds = model.predict()
# baseline with DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=1)
evaluation_model(model, x_train, y_train, x_valid, y_valid)
# baseline with RandomForestRegressor
model = RandomForestRegressor(random_state=1)
evaluation_model(model, x_train, y_train, x_valid, y_valid)
# n_estimators
n_est = [i for i in range(360, 500, 20)]
mae_scores= []
mse_scores = []
r2_scores = []
for i in n_est:
    model = RandomForestRegressor(n_estimators=i, random_state=1)
    mae_scores.append(evaluation_model(model, x_train, y_train, x_valid, y_valid)['mae'])
    mse_scores.append(evaluation_model(model, x_train, y_train, x_valid, y_valid)['mse'])
    r2_scores.append(evaluation_model(model, x_train, y_train, x_valid, y_valid)['r2'])
    
# plt.plot(n_est, mae_scores, label='mae score')
# plt.plot(n_est, mse_scores, label='mse score')
plt.plot(n_est, r2_scores, label='r2 score')
model = RandomForestRegressor(n_estimators=450, random_state=1)
evaluation_model(model, x_train, y_train, x_valid, y_valid)
# depth
depth_ = [i for i in range(8, 17)]
mae_scores= []
mse_scores = []
r2_scores = []
for i in depth_:
    model = RandomForestRegressor(n_estimators=450, max_depth=i, random_state=1)
    mae_scores.append(evaluation_model(model, x_train, y_train, x_valid, y_valid)['mae'])
    mse_scores.append(evaluation_model(model, x_train, y_train, x_valid, y_valid)['mse'])
    r2_scores.append(evaluation_model(model, x_train, y_train, x_valid, y_valid)['r2'])
    
# plt.plot(depth_, mae_scores, label='mae score')
# plt.plot(depth_, mse_scores, label='mse score')
plt.plot(depth_, r2_scores, label='r2 score')
model = RandomForestRegressor(n_estimators=450, max_depth=15, random_state=1)
evaluation_model(model, x_train, y_train, x_valid, y_valid)
model = RandomForestRegressor(n_estimators=450, max_depth=15, random_state=1)
score = {}
score['mae'] = np.mean(cross_val_score(model, train[corr_potential_features], train.SalePrice, cv=5, scoring="neg_mean_absolute_error"))
score['mse'] = np.mean(cross_val_score(model, train[corr_potential_features], train.SalePrice, cv=5, scoring="neg_mean_squared_error")) 
score['r2'] = np.mean(cross_val_score(model, train[corr_potential_features], train.SalePrice, cv=5))
score
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
y = train.SalePrice
X = train[corr_potential_features]
model_on_full_data = RandomForestRegressor(n_estimators=450, max_depth=15, random_state=1)
model_on_full_data = model_on_full_data.fit(X, y)
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
Id = test.Id
test = test[corr_potential_features]
test.fillna(0, inplace=True)
test_preds = model_on_full_data.predict(test)
output = pd.DataFrame({'Id': Id, 'SalePrice': test_preds})
output.to_csv('houseprice_test2.csv', index=False)
home_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col=0)
home_data
home_data.shape
home_data.select_dtypes(exclude=['O']).columns
len(home_data.select_dtypes(exclude=['O']).columns)
home_data.select_dtypes(exclude=['O']).describe().round(decimals=2)
home_data.select_dtypes(include=['O']).columns
len(home_data.select_dtypes(include=['O']).columns)
home_data.select_dtypes(include='O').describe()
target = home_data.SalePrice
plt.figure()
sns.distplot(target)
plt.title('Distribution of SalePrice')
plt.show()
sns.distplot(np.log(target))
plt.title('Distribution of Log-transfromed SalePrice')
plt.xlabel('log(SalePrice)')
plt.show()
print(f'Target has a skew of {str(target.skew().round(decimals=2))}, while the log-transofrmed SalePrice improves the skew to {np.log(target).skew().round(decimals=2)}')
num_attr = home_data.select_dtypes(exclude='object').drop('SalePrice', axis=1).copy()
fig = plt.figure(figsize=(15, 18))
skewness = {}
for i in range(len(num_attr.columns)):
    skewness[num_attr.columns[i]] = num_attr[num_attr.columns[i]].skew().round(decimals=2)
    fig.add_subplot(9, 4, i+1)
    try:
        sns.distplot(num_attr.iloc[:, i].dropna())
        plt.xlabel(num_attr.columns[i])
    except:
        plt.xlabel('ANOM_'+num_attr.columns[i])
        print(f'Anomalies in these columns {num_attr.columns[i]}')
        pass

plt.tight_layout()
plt.show()
for k,v in skewness.items():
    if v > 1:
        print(k, v)
fig = plt.figure(figsize=(12, 18))
for i in range(len(num_attr.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.boxplot(y=num_attr.iloc[:, i])

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(12, 18))
for i in range(len(num_attr.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.scatterplot(num_attr.iloc[:, i], target)

plt.tight_layout()
plt.show()
correlation = home_data.corr()
f, ax = plt.subplots(figsize=(14,12))
plt.title('Correlation of numerical attributes')
sns.heatmap(correlation)
plt.show()
correlation['SalePrice'].sort_values(ascending=False).head(15)
corr_columns = []
for i in correlation:
    corr_columns.append(i)
fig = plt.figure(figsize=(15, 18))
for i in range(len(correlation)):
    fig.add_subplot(9, 5, i+1)
    sns.scatterplot(home_data[corr_columns[i]], target)
    plt.title(f'Corr to SalePrice= {str(np.round(correlation.SalePrice[corr_columns[i]], decimals=3))}')
plt.tight_layout()
plt.show()
num_attr.isna().sum().sort_values(ascending=False).head()
cat_columns = home_data.select_dtypes(include='O').columns
cat_columns
fig = plt.figure(figsize=(18, 25))
for i in range(len(cat_columns)):
    fig.add_subplot(11, 4, i+1)
    sns.boxplot(home_data[cat_columns[i]], target)

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(18, 25))
for i in range(len(cat_columns)):
    fig.add_subplot(11, 4, i+1)
    sns.countplot(home_data[cat_columns[i]])

plt.tight_layout()
plt.show()
home_data[cat_columns].isna().sum().sort_values(ascending=False).head(17)
home_data_copy = home_data.copy()
col_fil_none = ['PoolQC',
 'MiscFeature',
 'Alley',
 'Fence',
 'FireplaceQu',
 'GarageCond',
 'GarageQual',
 'GarageFinish',
 'GarageType',
 'BsmtFinType2',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtQual',
 'BsmtCond', 'MasVnrType']
home_data_copy['MasVnrArea'] = home_data_copy.MasVnrArea.fillna(0)
for col in col_fil_none:
    home_data_copy[col] = home_data_copy[col].fillna('None')
home_data_copy.isna().sum().sort_values(ascending=False).head()
# home_data_copy = home_data_copy.drop(home_data_copy['LotFrontage']
#                                      [home_data_copy['LotFrontage']>200].index)
# home_data_copy = home_data_copy.drop(home_data_copy['LotArea']
#                                      [home_data_copy['LotArea']>100000].index)
# home_data_copy = home_data_copy.drop(home_data_copy['BsmtFinSF1']
#                                      [home_data_copy['BsmtFinSF1']>4000].index)
# home_data_copy = home_data_copy.drop(home_data_copy['TotalBsmtSF']
#                                      [home_data_copy['TotalBsmtSF']>6000].index)
# home_data_copy = home_data_copy.drop(home_data_copy['1stFlrSF']
#                                      [home_data_copy['1stFlrSF']>4000].index)
# home_data_copy = home_data_copy.drop(home_data_copy.GrLivArea
#                                      [(home_data_copy['GrLivArea']>4000) & 
#                                       (target<300000)].index)
# home_data_copy = home_data_copy.drop(home_data_copy.LowQualFinSF
#                                      [home_data_copy['LowQualFinSF']>550].index)
home_data_copy['LotFrontage'] = home_data_copy.drop(home_data_copy['LotFrontage'][home_data_copy['LotFrontage']>200].index)
home_data_copy['LotArea'] = home_data_copy.drop(home_data_copy['LotArea'][home_data_copy['LotArea']>100000].index)
home_data_copy['BsmtFinSF1'] = home_data_copy.drop(home_data_copy['BsmtFinSF1'][home_data_copy['BsmtFinSF1']>4000].index)
home_data_copy['BsmtFinSF2'] = home_data_copy.drop(home_data_copy['BsmtFinSF2'][home_data_copy['BsmtFinSF2']>1500].index)
home_data_copy['TotalBsmtSF'] = home_data_copy.drop(home_data_copy['TotalBsmtSF'][home_data_copy['TotalBsmtSF']>4000].index)
home_data_copy['1stFlrSF'] = home_data_copy.drop(home_data_copy['1stFlrSF'][home_data_copy['1stFlrSF']>4000].index)
home_data_copy['GrLivArea'] = home_data_copy.drop(home_data_copy['GrLivArea'][(home_data_copy['GrLivArea']>4000) & (target<300000)].index)
home_data_copy['WoodDeckSF'] = home_data_copy.drop(home_data_copy['WoodDeckSF'][home_data_copy['WoodDeckSF']>750].index)
home_data_copy['OpenPorchSF'] = home_data_copy.drop(home_data_copy['OpenPorchSF'][home_data_copy['OpenPorchSF']>400].index)
home_data_copy['EnclosedPorch'] = home_data_copy.drop(home_data_copy['EnclosedPorch'][home_data_copy['EnclosedPorch']>400].index)
home_data_copy['OpenPorchSF'] = home_data_copy.drop(home_data_copy['OpenPorchSF'][home_data_copy['OpenPorchSF']>400].index)
home_data_copy['MiscVal'] = home_data_copy.drop(home_data_copy['MiscVal'][home_data_copy['MiscVal']>5000].index)
home_data_copy.isna().sum().sort_values(ascending=False).head(15)
home_data_copy['SalePrice'] = np.log(home_data_copy['SalePrice'])
home_data_copy = home_data_copy.rename(columns={'SalePrice': 'SalePrice_log'})
home_data_copy
correlation['SalePrice'].sort_values(ascending=False)
home_data_copy.isna().sum().sort_values(ascending=False).head(15)
my_imputer = SimpleImputer(strategy='most_frequent')
imputed_home_data_copy = pd.DataFrame(my_imputer.fit_transform(home_data_copy), index=home_data_copy.index)
imputed_home_data_copy.columns = home_data_copy.columns
imputed_home_data_copy.isna().sum().sort_values(ascending=False).head(15)
col_drop = ['SalePrice_log', 'MSSubClass', 'YrSold', 'MiscVal', 'MoSold' , 'GarageArea', 'GarageYrBlt', 'YearRemodAdd']

# home_data_copy['LotFrontage'] = home_data_copy.LotFrontage.fillna(home_data_copy['LotFrontage'].mode()[0])
# home_data_copy['Electrical'] = home_data_copy.Electrical.fillna(home_data_copy['Electrical'].mode()[0])

X = imputed_home_data_copy.drop(col_drop, axis=1)
y = imputed_home_data_copy.SalePrice_log
X.info()
home_data_copy.select_dtypes(include='O')
X.isna().sum().sort_values(ascending=False).head(10)
# one-hot encoding to all categorical columns so far TODO: only with less than 10 unique values, greater than 10 deal with label encoder
X = pd.get_dummies(X)
x_train, x_valid, y_train, y_valid = train_test_split(X, y, random_state=1)
# my_imputer = SimpleImputer()
# imputed_x_train = pd.DataFrame(my_imputer.fit_transform(x_train), index=x_train.index)
# imputed_x_train.columns = x_train.columns
# imputed_x_valid = pd.DataFrame(my_imputer.transform(x_valid), index=x_valid.index)
# imputed_x_valid.columns = x_valid.columns
x_train.isna().sum().sort_values(ascending=False).head(10)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
mae_compare = pd.Series()
mae_compare.index.name = 'Algorithm'
model_1 = DecisionTreeRegressor(random_state=1)
mae_compare['DecisionTree'] = score(model_1, x_train, y_train, x_valid, y_valid)['mae']
model_2 = RandomForestRegressor(random_state=1)
mae_compare['RandomForest'] = score(model_2, x_train, y_train, x_valid, y_valid)['mae']
model_3 = LinearRegression()
mae_compare['LinearRegression'] = score(model_3, x_train, y_train, x_valid, y_valid)['mae']
model_4 = XGBRegressor()
mae_compare['XGBoost'] = score(model_4, x_train, y_train, x_valid, y_valid)['mae']
mae_compare.sort_values()
# imputer = SimpleImputer()
# imputed_X = pd.DataFrame(imputer.fit_transform(X), index=X.index)
# imputed_X.columns = X.columns
rmse_compare = pd.DataFrame({'Algorithm': [], 'RMSE': [], 'Error std': []})
rmse = np.sqrt(-cross_val_score(model_1, X, y, scoring='neg_mean_squared_error', cv=10))
rmse_compare.loc[0] = ('DecisionTree', rmse.mean(), rmse.std())
rmse = np.sqrt(-cross_val_score(model_2, X, y, scoring='neg_mean_squared_error', cv=10))
rmse_compare.loc[1] = ('RandomForest', rmse.mean(), rmse.std())
rmse = np.sqrt(-cross_val_score(model_3, X, y, scoring='neg_mean_squared_error', cv=10))
rmse_compare.loc[2] = ('LinearRegression', rmse.mean(), rmse.std())
rmse = np.sqrt(-cross_val_score(model_4, X, y, scoring='neg_mean_squared_error', cv=10))
rmse_compare.loc[3] = ('XGBoost', rmse.mean(), rmse.std())
rmse_compare.sort_values(by='RMSE')
# Tuning RandomForestRegressor
# param_grid = [{'n_estimators': [50, 100, 150, 200, 250], 'max_depth': [2, 4, 8, 10]}]
# top_reg = XGBRegressor()

# # -------------------------------------------------------
# grid_search = GridSearchCV(top_reg, param_grid, cv=5, 
#                            scoring='neg_mean_squared_error')

# grid_search.fit(imputed_X, y)

# grid_search.best_params_
# n_estimators
n_est = [i for i in range(25, 100, 10)]
mae_scores= []
mse_scores = []
for i in n_est:
    model = RandomForestRegressor(n_estimators=i, random_state=1)
    mae_scores.append(score(model, x_train, y_train, x_valid, y_valid)['mae'])
    mse_scores.append(score(model, x_train, y_train, x_valid, y_valid)['mse'])
    
plt.plot(n_est, mae_scores, label='mae score')
# plt.plot(n_est, mse_scores, label='mse score')
# depth
depth_ = [i for i in range(12, 24)]
mae_scores= []
mse_scores = []
for i in depth_:
    model = RandomForestRegressor(n_estimators=55, max_depth=i, random_state=1)
    mae_scores.append(score(model, x_train, y_train, x_valid, y_valid)['mae'])
    mse_scores.append(score(model, x_train, y_train, x_valid, y_valid)['mse'])
    
plt.plot(depth_, mae_scores, label='mae score')
# plt.plot(depth_, mse_scores, label='mse score')
model = RandomForestRegressor(n_estimators=55, random_state=1)
score(model, x_train, y_train, imputed_x_valid, y_valid)
model = RandomForestRegressor(n_estimators=55, max_depth=19, random_state=1)
score(model, x_train, y_train, imputed_x_valid, y_valid)
model = LinearRegression()
score(model, x_train, y_train, imputed_x_valid, y_valid)
home_data.isna().sum().sort_values(ascending=False).head(20)
test_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col=0)
test_X = test_data.copy()

for col in col_fil_none:
    test_X[col] = test_X[col].fillna('None')

if 'SalePrice_log' in col_drop:
    col_drop.remove('SalePrice_log')

test_X = test_data.drop(col_drop, axis=1)
test_X['LotFrontage'] = test_X['LotFrontage'].fillna(test_X['LotFrontage'].mode()[0])
test_X['MasVnrArea'] = test_X['MasVnrArea'].fillna(0)
test_X['BsmtHalfBath'] = test_X['BsmtHalfBath'].fillna(test_X['BsmtHalfBath'].mode()[0])
test_X['BsmtFullBath'] = test_X['BsmtFullBath'].fillna(test_X['BsmtFullBath'].mode()[0])
test_X['GarageCars'] = test_X['GarageCars'].fillna(test_X['GarageCars'].mode()[0])
test_X['BsmtFinSF1'] = test_X['BsmtFinSF1'].fillna(test_X['BsmtFinSF1'].mode()[0])
test_X['BsmtFinSF2'] = test_X['BsmtFinSF2'].fillna(test_X['BsmtFinSF2'].mode()[0])
test_X['BsmtUnfSF'] = test_X['BsmtUnfSF'].fillna(test_X['BsmtUnfSF'].mode()[0])
test_X['TotalBsmtSF'] = test_X['TotalBsmtSF'].fillna(test_X['TotalBsmtSF'].mode()[0])

test_X = pd.get_dummies(test_X)
# final_train, final_test = X.align(test_X, join='left', axis=1)
# final_test_imputed = pd.DataFrame(my_imputer.transform(test_X), index=test_X.index)
# final_test_imputed.columns = final_test.columns
# final_train_imputed = pd.DataFrame(my_imputer.fit_transform(final_train), index=final_train.index)
# final_train_imputed.columns = final_train.columns

final_model = RandomForestRegressor(n_estimators=55, max_depth=19, random_state=1)
drop_x_train = X.drop([i for i in X.columns if i not in test_X.columns], axis=1)
final_model.fit(drop_x_train, y)

test_preds = final_model.predict(test_X)
output = pd.DataFrame({'Id': test_data.index, 'SalePrice': inv_y(test_preds)})
output.to_csv('houseprice_test5.csv', index=False)
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col=0)

train.shape, test.shape
train.isna().sum().sort_values(ascending=False).head(20)
train.select_dtypes(exclude='O').isna().sum().sort_values(ascending=False).head()
train.select_dtypes(include='O').isna().sum().sort_values(ascending=False).head(20)
missing_col = ['PoolQC',
 'MiscFeature',
 'Alley',
 'Fence',
 'FireplaceQu',
 'LotFrontage',
 'GarageType',
 'GarageCond',
 'GarageFinish',
 'GarageQual',
 'GarageYrBlt',
 'BsmtFinType2',
 'BsmtExposure',
 'BsmtQual',
 'BsmtCond',
 'BsmtFinType1',
 'MasVnrArea',
 'MasVnrType',
 'Electrical']

missing_num_col = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']
missing_cat_col = [i for i in missing_col if i not in missing_num_col]
train_copy = train.copy()
test_copy = test.copy()

target = train.SalePrice
def drop_outliers(df):
    '''drop all outstanding rows, which leads more columns with missing values'''
    df['LotFrontage'] = df['LotFrontage'].drop(df['LotFrontage'][df['LotFrontage']>200].index)
    df['LotArea'] = df['LotArea'].drop(df['LotArea'][df['LotArea']>100000].index)
    df['BsmtFinSF1'] = df['BsmtFinSF1'].drop(df['BsmtFinSF1'][df['BsmtFinSF1']>4000].index)
#     df['BsmtFinSF2'] = df['BsmtFinSF2'].drop(df['BsmtFinSF2'][df['BsmtFinSF2']>1500].index)
    df['TotalBsmtSF'] = df['TotalBsmtSF'].drop(df['TotalBsmtSF'][df['TotalBsmtSF']>5000].index)
#     df['1stFlrSF'] = df['1stFlrSF'].drop(df['1stFlrSF'][df['1stFlrSF']>4000].index)
    df['GrLivArea'] = df['GrLivArea'].drop(df['GrLivArea'][df['GrLivArea']>4000].index)

#     df['GrLivArea'] = df['GrLivArea'].drop(df['GrLivArea'][(df['GrLivArea']>4000) & (target<300000)].index)
#     df['LowQualFinSF'] = df['LowQualFinSF'].drop(df['LowQualFinSF'][df['LowQualFinSF']>550].index)
#     df['WoodDeckSF'] = df['WoodDeckSF'].drop(df['WoodDeckSF'][df['WoodDeckSF']>750].index)
#     df['OpenPorchSF'] = df['OpenPorchSF'].drop(df['OpenPorchSF'][df['OpenPorchSF']>400].index)
#     df['EnclosedPorch'] = df['EnclosedPorch'].drop(df['EnclosedPorch'][df['EnclosedPorch']>400].index)
#     df['OpenPorchSF'] = df['OpenPorchSF'].drop(df['OpenPorchSF'][df['OpenPorchSF']>400].index)
#     df['MiscVal'] = df['MiscVal'].drop(df['MiscVal'][df['MiscVal']>5000].index)
    return df
train_copy = drop_outliers(train_copy)
# test_copy = drop_outliers(test_copy)

train_copy
train_copy.isna().sum().sort_values(ascending=False).head(30)
missing_col_2 = ['PoolQC',
 'MiscFeature',
 'Alley',
 'Fence',
 'FireplaceQu',
 'LotFrontage',
 'GarageCond',
 'GarageQual',
 'GarageType',
 'GarageYrBlt',
 'GarageFinish',
 'BsmtExposure',
 'BsmtFinType2',
 'BsmtQual',
 'BsmtCond',
 'BsmtFinType1',
 'MasVnrType',
 'MasVnrArea',
 'OpenPorchSF',
 'LotArea',
 'GrLivArea',
 'MiscVal',
 'TotalBsmtSF',
 'LowQualFinSF',
 '1stFlrSF',
 'Electrical',
 'WoodDeckSF',
 'BsmtFinSF1',
 'EnclosedPorch']

new_missing_num_col = [i for i in missing_col_2 if i not in missing_col]
new_missing_num_col
train_copy.describe(include='O')
drop_col = ['PoolQC', 'MiscFeature', 'OverallCond', 
            'MSSubClass', 'YrSold', 'MiscVal', 'MoSold' , 'GarageArea', 'GarageYrBlt', 'GarageCond', 'YearRemodAdd',
            'LotShape','Condition2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2',
             'Alley', 'FireplaceQu','Fence', 'LotConfig', 'Condition2', 'RoofStyle', 'Exterior2nd', 'MasVnrType', 'ExterCond', 'BsmtCond']
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col=0)
target = train.SalePrice
train_copy = train.copy()
test_copy = test.copy()
highly_corr = ['GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageCars']
many_missing_val = ['PoolQC','MiscFeature','Alley']
less_corr_target = ['MoSold','YrSold']

col = train.columns
most_one_value = []  # ['Street', 'Utilities','Condition2','RoofMatl','Heating','LowQualFinSF','3SsnPorch','PoolArea','MiscVal']
for i in col:
    counts = train[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(train) * 100 > 96:
        most_one_value.append(i)
        
total_drop_cols = highly_corr + many_missing_val + less_corr_target + most_one_value
train_copy = train_copy.drop(total_drop_cols, axis=1)
test_copy = test_copy.drop(total_drop_cols, axis=1)
train_copy.shape, test_copy.shape
train_copy = drop_outliers(train_copy)
check_null(train_copy, 20)
X = train_copy.loc[:, train_copy.columns!='SalePrice']
log_target = np.log(target)
x_train, x_valid, y_train, y_valid = train_test_split(X, log_target, random_state=1)
cat = ['GarageType','GarageFinish','BsmtFinType2','BsmtExposure','BsmtFinType1', 
       'GarageCond','GarageQual','BsmtCond','BsmtQual','FireplaceQu','Fence',"KitchenQual",
       "HeatingQC",'ExterQual','ExterCond']
x_train[cat] = x_train[cat].fillna("NA")
x_valid[cat] = x_valid[cat].fillna("NA")
# train_copy[cat] = train_copy[cat].fillna("NA")
test_copy[cat] = test_copy[cat].fillna("NA")

cols = ["MasVnrType", "MSZoning", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "Functional", "GrLivArea"]
x_train[cols] = x_train.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))
x_valid[cols] = x_valid.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))
# train_copy[cols] = train_copy.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))
test_copy[cols] = test_copy.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))

#for correlated relationship
x_train['LotArea'] = x_train.groupby('Neighborhood')['LotArea'].transform(lambda x: x.fillna(x.mean()))
x_train['LotFrontage'] = x_train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
x_train['GarageArea'] = x_train.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))
x_train['MSZoning'] = x_train.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
x_valid['LotArea'] = x_valid.groupby('Neighborhood')['LotArea'].transform(lambda x: x.fillna(x.mean()))
x_valid['LotFrontage'] = x_valid.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
x_valid['GarageArea'] = x_valid.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))
x_valid['MSZoning'] = x_valid.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# train_copy['LotArea'] = train_copy.groupby('Neighborhood')['LotArea'].transform(lambda x: x.fillna(x.mean()))
# train_copy['LotFrontage'] = train_copy.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
# train_copy['GarageArea'] = train_copy.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))
# train_copy['MSZoning'] = train_copy.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
test_copy['LotArea'] = test_copy.groupby('Neighborhood')['LotArea'].transform(lambda x: x.fillna(x.mean()))
test_copy['LotFrontage'] = test_copy.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
test_copy['GarageArea'] = test_copy.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.mean()))
test_copy['MSZoning'] = test_copy.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

#numerical
cont = ["BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea"]
x_train[cont] = x_train[cont].fillna(x_train[cont].mean())
x_valid[cont] = x_valid[cont].fillna(x_valid[cont].mean())
# train_copy[cont] = train_copy[cont].fillna(train_copy[cont].mean())
test_copy[cont] = test_copy[cont].fillna(test_copy[cont].mean())

print(check_null(x_train, 5))
print(check_null(x_valid, 5))
# print(check_null(train_copy, 5))
print(check_null(test_copy, 5))
x_train['MSSubClass'] = x_train['MSSubClass'].apply(str)
x_valid['MSSubClass'] = x_valid['MSSubClass'].apply(str)
# train_copy['MSSubClass'] = train_copy['MSSubClass'].apply(str)
test_copy['MSSubClass'] = test_copy['MSSubClass'].apply(str)
# Mapping Ordinal Features
ordinal_map = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}
fintype_map = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}
expose_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
fence_map = {'GdPrv': 4,'MnPrv': 3,'GdWo': 2, 'MnWw': 1,'NA': 0}
ord_col = ['ExterQual','ExterCond','BsmtQual', 'BsmtCond','HeatingQC','KitchenQual','GarageQual','GarageCond', 'FireplaceQu']
for col in ord_col:
    x_train[col] = x_train[col].map(ordinal_map)
    x_valid[col] = x_valid[col].map(ordinal_map)
#     train_copy[col] = train_copy[col].map(ordinal_map)
    test_copy[col] = test_copy[col].map(ordinal_map)
    
fin_col = ['BsmtFinType1','BsmtFinType2']
for col in fin_col:
    x_train[col] = x_train[col].map(fintype_map)
    x_valid[col] = x_valid[col].map(fintype_map)
#     train_copy[col] = train_copy[col].map(fintype_map)
    test_copy[col] = test_copy[col].map(fintype_map)

x_train['BsmtExposure'] = x_train['BsmtExposure'].map(expose_map)
x_train['Fence'] = x_train['Fence'].map(fence_map)
x_valid['BsmtExposure'] = x_valid['BsmtExposure'].map(expose_map)
x_valid['Fence'] = x_valid['Fence'].map(fence_map)
# train_copy['BsmtExposure'] = train_copy['BsmtExposure'].map(expose_map)
# train_copy['Fence'] = train_copy['Fence'].map(fence_map)
test_copy['BsmtExposure'] = test_copy['BsmtExposure'].map(expose_map)
test_copy['Fence'] = test_copy['Fence'].map(fence_map)
# feature engineer
x_train['TotalLot'] = x_train['LotFrontage'] + x_train['LotArea']
x_train['TotalBsmtFin'] = x_train['BsmtFinSF1'] + x_train['BsmtFinSF2']
x_train['TotalSF'] = x_train['TotalBsmtSF'] + x_train['2ndFlrSF']
x_train['TotalBath'] = x_train['FullBath'] + x_train['HalfBath']
x_train['TotalPorch'] = x_train['OpenPorchSF'] + x_train['EnclosedPorch'] + x_train['ScreenPorch']
x_valid['TotalLot'] = x_valid['LotFrontage'] + x_valid['LotArea']
x_valid['TotalBsmtFin'] = x_valid['BsmtFinSF1'] + x_valid['BsmtFinSF2']
x_valid['TotalSF'] = x_valid['TotalBsmtSF'] + x_valid['2ndFlrSF']
x_valid['TotalBath'] = x_valid['FullBath'] + x_valid['HalfBath']
x_valid['TotalPorch'] = x_valid['OpenPorchSF'] + x_valid['EnclosedPorch'] + x_valid['ScreenPorch']
# train_copy['TotalLot'] = train_copy['LotFrontage'] + train_copy['LotArea']
# train_copy['TotalBsmtFin'] = train_copy['BsmtFinSF1'] + train_copy['BsmtFinSF2']
# train_copy['TotalSF'] = train_copy['TotalBsmtSF'] + train_copy['2ndFlrSF']
# train_copy['TotalBath'] = train_copy['FullBath'] + train_copy['HalfBath']
# train_copy['TotalPorch'] = train_copy['OpenPorchSF'] + train_copy['EnclosedPorch'] + train_copy['ScreenPorch']
test_copy['TotalLot'] = test_copy['LotFrontage'] + test_copy['LotArea']
test_copy['TotalBsmtFin'] = test_copy['BsmtFinSF1'] + test_copy['BsmtFinSF2']
test_copy['TotalSF'] = test_copy['TotalBsmtSF'] + test_copy['2ndFlrSF']
test_copy['TotalBath'] = test_copy['FullBath'] + test_copy['HalfBath']
test_copy['TotalPorch'] = test_copy['OpenPorchSF'] + test_copy['EnclosedPorch'] + test_copy['ScreenPorch']

colum = ['MasVnrArea','TotalBsmtFin','TotalBsmtSF','2ndFlrSF','WoodDeckSF','TotalPorch']
for col in colum:
    col_name = col+'_bin'
    x_train[col_name] = x_train[col].apply(lambda x: 1 if x > 0 else 0)
    x_valid[col_name] = x_valid[col].apply(lambda x: 1 if x > 0 else 0)
#     train_copy[col_name] = train_copy[col].apply(lambda x: 1 if x > 0 else 0)
    test_copy[col_name] = test_copy[col].apply(lambda x: 1 if x > 0 else 0)
    
x_train.shape, x_valid.shape, test_copy.shape
remaining_cat_features = x_train.select_dtypes(include='O').columns
# remaining_cat_features = train_copy.select_dtypes(include='O').columns
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_encoder.fit(x_train[remaining_cat_features])
# OH_encoder.fit(train_copy[remaining_cat_features])
column_name = OH_encoder.get_feature_names()

OH_col_x_train = pd.DataFrame(OH_encoder.transform(x_train[remaining_cat_features]), columns=column_name, index=x_train.index)
OH_col_x_valid = pd.DataFrame(OH_encoder.transform(x_valid[remaining_cat_features]), columns=column_name, index=x_valid.index)
# OH_col_train = pd.DataFrame(OH_encoder.transform(train_copy[remaining_cat_features]), columns=column_name, index=train_copy.index)
OH_col_test = pd.DataFrame(OH_encoder.transform(test_copy[remaining_cat_features]), columns=column_name, index=test_copy.index)

num_x_train = x_train.drop(remaining_cat_features, axis=1)
num_x_valid = x_valid.drop(remaining_cat_features, axis=1)
# num_train = train_copy.drop(remaining_cat_features, axis=1)
num_test = test_copy.drop(remaining_cat_features, axis=1)

OH_x_train = pd.concat([num_x_train, OH_col_x_train], axis=1)
OH_x_valid = pd.concat([num_x_valid, OH_col_x_valid], axis=1)
# OH_train = pd.concat([num_train, OH_col_train], axis=1)
OH_test = pd.concat([num_test, OH_col_test], axis=1) 
OH_x_train.shape, OH_x_valid.shape, OH_test.shape
# train_copy = pd.get_dummies(train_copy)
# test_copy = pd.get_dummies(test_copy)
X = OH_train.loc[:, OH_train.columns!='SalePrice']
log_target = np.log(target)
if X.shape[1] != OH_test.shape[1]:
    col_not_test = [i for i in X.columns if i not in OH_test.columns]
    col_not_train = [i for i in OH_test.columns if i not in X.columns]
    drop_cols = col_not_test + col_not_train
    print('Start dropping...')
    X = X.drop(col_not_test, axis=1)
    OH_test = OH_test.drop(col_not_train, axis=1)
    
X.shape, OH_test.shape
x_train, x_valid, y_train, y_valid = train_test_split(X, log_target, test_size=0.4, random_state=1)
# n_estimators
n_est = [i for i in range(90, 300, 10)]
mae_scores= []
mse_scores = []
for i in n_est:
    model = RandomForestRegressor(n_estimators=i, random_state=1)
    mae_scores.append(score(model, OH_x_train, y_train, OH_x_valid, y_valid)['mae'])
    mse_scores.append(score(model, OH_x_train, y_train, OH_x_valid, y_valid)['mse'])
    
plt.plot(n_est, mae_scores, label='mae score')
# depth
depth_ = [i for i in range(23,35)]
mae_scores= []
mse_scores = []
for i in depth_:
    model = RandomForestRegressor(n_estimators=80, max_depth=i, random_state=1)
    mae_scores.append(score(model, OH_x_train, y_train, OH_x_valid, y_valid)['mae'])
    mse_scores.append(score(model, OH_x_train, y_train, OH_x_valid, y_valid)['mse'])
    
plt.plot(depth_, mae_scores, label='mae score')
# plt.plot(depth_, mse_scores, label='mse score')
model = RandomForestRegressor(n_estimators=80, max_depth=25, random_state=1)
score(model, OH_x_train, y_train, OH_x_valid, y_valid)
model = RandomForestRegressor(n_estimators=80, max_depth=14, random_state=1)
score(model, OH_x_train, y_train, OH_x_valid, y_valid)
X.shape, log_target.shape, OH_test.shape
total_x = pd.concat([OH_x_train,OH_x_valid]).sort_index()
total_x.shape, log_target.shape, OH_test.shape
final_model = RandomForestRegressor(n_estimators=80, max_depth=14, random_state=1)
final_model.fit(OH_x_train, y_train)
preds = final_model.predict(OH_test)
output = pd.DataFrame({'Id': test.index, 'SalePrice': inv_y(preds)})
output.to_csv('houseprice_test18.csv', index=False)
train_copy['GarageYrBlt'] = train_copy['GarageYrBlt'].fillna(0)
train_copy['MasVnrArea'] = train_copy['MasVnrArea'].fillna(0)
test_copy['GarageYrBlt'] = test_copy['GarageYrBlt'].fillna(0)
test_copy['MasVnrArea'] = test_copy['MasVnrArea'].fillna(0)

train_copy = train_copy.drop(drop_col, axis=1)
test_copy = test_copy.drop(drop_col, axis=1)

train_copy.shape, test_copy.shape
# can be done after split dataset or before k-fold cross-val
missing_col_fil_none = list(set(missing_cat_col) - set([i for i in missing_cat_col if i in drop_col]+['Electrical']))
missing_col_fil_mode = ['Electrical']
missing_col_fil_median = list(set(new_missing_num_col)-set([i for i in drop_col if i in new_missing_num_col])) +['LotFrontage']
X = train_copy.loc[:, train_copy.columns != 'SalePrice']
trans_target = np.log(target)
x_train, x_valid, y_train, y_valid = train_test_split(X, trans_target, random_state=1)
x_train.shape, x_valid.shape
imputed_none_x_train = x_train.copy()
imputed_none_x_valid = x_valid.copy()
imputed_none_test = test_copy.copy()
for col in missing_col_fil_none:
    imputed_none_x_train[col] = x_train[col].fillna('None')
    imputed_none_x_valid[col] = x_valid[col].fillna('None')
    imputed_none_test[col] = test_copy[col].fillna('None')
imputed_mode_x_train = imputed_none_x_train.copy()
imputed_mode_x_valid = imputed_none_x_valid.copy()
imputed_mode_test = imputed_none_test.copy()
for col in missing_col_fil_mode:
    imputed_mode_x_train[col] = imputed_none_x_train[col].fillna(imputed_none_x_train[col].mode()[0])
    imputed_mode_x_valid[col] = imputed_none_x_valid[col].fillna(imputed_none_x_valid[col].mode()[0])
    imputed_mode_test[col] = imputed_mode_test[col].fillna(imputed_mode_test[col].mode()[0])
imputed_median_x_train = imputed_mode_x_train.copy()
imputed_median_x_valid = imputed_mode_x_valid.copy()
imputed_median_test = imputed_mode_test.copy()
for col in missing_col_fil_median:
    imputed_median_x_train[col] = imputed_mode_x_train[col].fillna(imputed_mode_x_train[col].mean())
    imputed_median_x_valid[col] = imputed_mode_x_valid[col].fillna(imputed_mode_x_valid[col].mean())
    imputed_median_test[col] = imputed_median_test[col].fillna(imputed_median_test[col].mean())
check_null(imputed_median_x_train).head()
# check_null(imputed_median_test)
test_imputer = SimpleImputer(strategy='most_frequent')
imputed_median_test = pd.DataFrame(test_imputer.fit_transform(imputed_median_test), index=imputed_median_test.index, columns=imputed_median_test.columns)
train_copy.describe(include='O')
# ordinal_features = ['MSZoning','Neighborhood', 'Condition1', 'HouseStyle', 'Functional',
#                     'ExterQual', 'BsmtQual','KitchenQual', 'FireplaceQu', 'GarageQual', 'SaleType']
cat_features = [ 'Neighborhood', 'RoofMatl', 'Exterior1st', 'HeatingQC']
remaining_cat_features = list(set(imputed_median_x_train.select_dtypes(include='O').columns) - set(ordinal_features))
ordinal_features = list(set(imputed_median_x_train.select_dtypes(include='O').columns) - set(cat_features))
label_x_train = imputed_median_x_train.copy()
label_x_valid = imputed_median_x_valid.copy()
label_test = imputed_median_test.copy()

label_encoder = LabelEncoder()
for col in ordinal_features:
    label_x_train[col] = label_encoder.fit_transform(imputed_median_x_train[col])
    label_x_valid[col] = label_encoder.transform(imputed_median_x_valid[col])
    label_test[col] = label_encoder.transform(imputed_median_test[col])
label_x_train.shape, label_test.shape
remaining_cat_features = cat_features
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_encoder.fit(label_x_train[remaining_cat_features])
column_name = OH_encoder.get_feature_names()
OH_col_train = pd.DataFrame(OH_encoder.transform(label_x_train[remaining_cat_features]), columns=column_name, index=label_x_train.index)
OH_col_valid = pd.DataFrame(OH_encoder.transform(label_x_valid[remaining_cat_features]), columns=column_name, index=label_x_valid.index)
OH_col_test = pd.DataFrame(OH_encoder.transform(label_test[remaining_cat_features]), columns=column_name, index=label_test.index)
num_x_train = label_x_train.drop(remaining_cat_features, axis=1)
num_x_valid = label_x_valid.drop(remaining_cat_features, axis=1)
num_test = label_test.drop(remaining_cat_features, axis=1)
OH_x_train = pd.concat([num_x_train, OH_col_train], axis=1)
OH_x_valid = pd.concat([num_x_valid, OH_col_valid], axis=1)
OH_test = pd.concat([num_test, OH_col_test], axis=1) 
OH_x_train.shape, OH_x_valid.shape, OH_test.shape
# n_estimators
n_est = [i for i in range(50, 150, 10)]
mae_scores= []
mse_scores = []
for i in n_est:
    model = RandomForestRegressor(n_estimators=i, random_state=1)
    mae_scores.append(score(model, OH_x_train, y_train, OH_x_valid, y_valid)['mae'])
    mse_scores.append(score(model, OH_x_train, y_train, OH_x_valid, y_valid)['mse'])
    
plt.plot(n_est, mae_scores, label='mae score')
# depth
depth_ = [i for i in range(5, 20)]
mae_scores= []
mse_scores = []
for i in depth_:
    model = RandomForestRegressor(n_estimators=80, max_depth=i, random_state=1)
    mae_scores.append(score(model, OH_x_train, y_train, OH_x_valid, y_valid)['mae'])
    mse_scores.append(score(model, OH_x_train, y_train, OH_x_valid, y_valid)['mse'])
    
plt.plot(depth_, mae_scores, label='mae score')
# plt.plot(depth_, mse_scores, label='mse score')
model = RandomForestRegressor(n_estimators=80, random_state=1)
score(model, OH_x_train, y_train, OH_x_valid, y_valid)
total_x = pd.concat([OH_x_train, OH_x_valid])
total_y = pd.concat([y_train, y_valid])
total_x.shape, OH_test.shape
final_model = RandomForestRegressor(n_estimators=80, random_state=1)
final_model.fit(total_x, total_y)
preds = final_model.predict(OH_test)
output = pd.DataFrame({'Id': test.index, 'SalePrice': inv_y(preds)})
output.to_csv('houseprice_test12.csv', index=False)
