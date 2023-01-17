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
test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv",index_col='Id')
train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv",index_col='Id')

print(test.shape)
print(train.shape)
test.info()
train.info()
import matplotlib

missing1 = train.isnull().sum()
missing1 = missing1[missing1>0]
missing1.sort_values()
missing1.plot.bar()

missing1
missing2 = test.isnull().sum()
missing2 = missing2[missing2>0]
missing2.sort_values()
missing2.plot.bar()

missing2

numerical_data = train.select_dtypes(exclude=['object']).drop(['SalePrice'],axis=1).copy()
print(numerical_data.columns)
categorical_data = train.select_dtypes(include = ['object']).copy()
print(categorical_data.columns)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
fig = plt.figure(figsize=(12,18))
for i in range(len(numerical_data.columns)):
    fig.add_subplot(9,4,i+1)
    sns.distplot(numerical_data.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})
    plt.xlabel(numerical_data.columns[i])
plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(12,18))
for i in range(len(numerical_data.columns)):
    fig.add_subplot(9,4,i+1)
    sns.boxplot(y=numerical_data.iloc[:,i])

plt.tight_layout()
plt.show()
fig3 = plt.figure(figsize=(12,18))
for i in range(len(numerical_data.columns)):
    fig3.add_subplot(9, 4, i+1)
    sns.scatterplot(numerical_data.iloc[:, i],train['SalePrice'])
plt.tight_layout()
plt.show()
sns.regplot(train['LotFrontage'],train['SalePrice'])

train = train.drop(train[train['LotFrontage']>200].index)
train = train.drop(train[train['LotArea']>100000].index)
train = train.drop(train[train['MasVnrArea']>1200].index)
train = train.drop(train[train['BsmtFinSF1']>4000].index)
train = train.drop(train[train['TotalBsmtSF']>4000].index)
train = train.drop(train[train['1stFlrSF']>4000].index)
train = train.drop(train[train['EnclosedPorch']>500].index)
train = train.drop(train[train['MiscVal']>5000].index)
train = train.drop(train[train['BsmtFinSF1']>4000].index)
train = train.drop(train[train['WoodDeckSF']>800].index)
train = train.drop(train[train['BsmtFinSF1']>4000].index)
train = train.drop(train[(train['LowQualFinSF']>600) & (train['SalePrice']>400000)].index)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
numerical_corelation = train.select_dtypes(exclude='object').corr()
plt.figure(figsize=(20,20))
plt.title("High Corelation")
sns.heatmap(numerical_corelation>0.8, annot=True, square=True)

train.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True)
test.drop(columns=['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF'],axis=1,inplace=True)
train=train.drop(columns=['Street','Utilities']) 
test=test.drop(columns=['Street','Utilities'])
train.isnull().mean().sort_values(ascending=False).head(5)
train.drop(columns=['Alley','MiscFeature','PoolQC','PoolArea'], axis=1, inplace=True)
test.drop(columns=['Alley','MiscFeature','PoolQC','PoolArea'], axis=1, inplace=True)
#look at the percentage of each data missing 
null = pd.DataFrame(data={'Train Null Percentage': train.isnull().sum()[train.isnull().sum() > 0], 
'Test Null Percentage': test.isnull().sum()[test.isnull().sum() > 0]})
null = (null/len(train)) * 100

null.index.name='Feature'
null
home_num_features = train.select_dtypes(exclude='object').isnull().mean()
test_num_features = test.select_dtypes(exclude='object').isnull().mean()

num_null_features = pd.DataFrame(data={'Missing Num Train Percentage: ': home_num_features[home_num_features>0]*100, 'Missing Num Test Percentage: ': test_num_features[test_num_features>0]*100})
num_null_features.index.name = 'Numerical Features'
num_null_features
for df in [train, test]:
    for col in ('GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 
                'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotalBsmtSF',
                'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal',
                'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea'):
                    df[col] = df[col].fillna(0)

_=sns.regplot(train['LotFrontage'],train['SalePrice'])
home_num_features = train.select_dtypes(exclude='object').isnull().mean()
test_num_features = test.select_dtypes(exclude='object').isnull().mean()

num_null_features = pd.DataFrame(data={'Missing Num Home Percentage: ': home_num_features[home_num_features>0]*100, 'Missing Num Test Percentage: ': test_num_features[test_num_features>0]*100})
num_null_features.index.name = 'Numerical Features'
num_null_features
cat_col = train.select_dtypes(include='object').columns
print(cat_col)

home_cat_features = train.select_dtypes(include='object').isnull().mean()
test_cat_features = test.select_dtypes(include='object').isnull().mean()

cat_null_features = pd.DataFrame(data={'Missing Cat Home Percentage: ': home_cat_features[home_cat_features>0]*100, 'Missing Cat Test Percentage: ': test_cat_features[test_cat_features>0]*100})
cat_null_features.index.name = 'Categorical Features'
cat_null_features
cat_col = train.select_dtypes(include='object').columns

columns = (len(cat_col)/4)+1

fg, ax = plt.subplots(figsize=(20, 30))

for i, col in enumerate(cat_col):
    fg.add_subplot(columns, 4, i+1)
    sns.countplot(train[col])
    plt.xlabel(col)
    plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
for df in [train, test]:
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                  'BsmtFinType2', 'Neighborhood', 'BldgType', 'HouseStyle', 'MasVnrType', 'FireplaceQu', 'Fence'):
        df[col] = df[col].fillna('None')
for df in [train, test]:
    for col in ('LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Condition1', 'RoofStyle',
                  'Electrical', 'Functional', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'ExterQual', 'ExterCond',
                  'Foundation', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'PavedDrive', 'SaleType', 'SaleCondition'):
        df[col] = df[col].fillna(df[col].mode()[0])                            #returns mode of each column if no value is passed ,else if axis = 1 ,then we may return mode of row instead.
home_cat_features = train.select_dtypes(include='object').isnull().mean()
test_cat_features = test.select_dtypes(include='object').isnull().mean()

cat_null_features = pd.DataFrame(data={'Missing Cat Home Percentage: ': home_cat_features[home_cat_features>0]*100, 'Missing Cat Test Percentage: ': test_cat_features[test_cat_features>0]*100})
cat_null_features.index.name = 'Categorical Features'
cat_null_features
sns.regplot(train['LotFrontage'],train['SalePrice'])
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
train.corr()['LotFrontage'].sort_values(ascending=False)
train.corr()['SalePrice'].sort_values(ascending=False)
train.isnull().sum().sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
train['MSSubClass'] = train['MSSubClass'].apply(str)
train['MSSubClass']

train['MSSubClass'] = train['MSSubClass'].apply(str)
test['MSSubClass'] = test['MSSubClass'].apply(str)

train['MoSold'] = train['MoSold'].apply(str)
test['MoSold'] = test['MoSold'].apply(str)

train['YrSold'] = train['MoSold'].apply(str)
test['YrSold'] = test['MoSold'].apply(str)
train['MSZoning'] = train.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
test['MSZoning'] = test.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
test.isnull().sum().sort_values(ascending=False)
train['TotalSF']=train['TotalBsmtSF']  + train['2ndFlrSF']
test['TotalSF']=test['TotalBsmtSF']  + test['2ndFlrSF']
train['TotalBath']= train['BsmtFullBath'] + train['FullBath'] + (0.5*train['BsmtHalfBath']) + (0.5*train['HalfBath'])
test['TotalBath']=test['BsmtFullBath'] + test['FullBath'] + 0.5*test['BsmtHalfBath'] + 0.5*test['HalfBath']
train['YrBltAndRemod']=train['YearBuilt']+(train['YearRemodAdd']/2)
test['YrBltAndRemod']=test['YearBuilt']+(test['YearRemodAdd']/2)
train['Porch_SF'] = (train['OpenPorchSF'] + train['3SsnPorch'] + train['EnclosedPorch'] + train['ScreenPorch'] + train['WoodDeckSF'])
test['Porch_SF'] = (test['OpenPorchSF'] + test['3SsnPorch'] + test['EnclosedPorch'] + test['ScreenPorch'] + test['WoodDeckSF'])
train['Has2ndfloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasBsmt'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasFirePlace'] = train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
train['Has2ndFlr']=train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasBsmt']=train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

test['Has2ndfloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasBsmt'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasFirePlace'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
test['Has2ndFlr']=test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasBsmt']=test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)


print(type(train['LotArea'][10]))

train['LotArea'] = train['LotArea'].astype(np.int64)
test['LotArea'] = test['LotArea'].astype(np.int64)
train['MasVnrArea'] = train['MasVnrArea'].astype(np.int64)
test['MasVnrArea'] = test['MasVnrArea'].astype(np.int64)

print ("Skew of SalePrice:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='yellow')
plt.show()
print ("Skew of Log-Transformed SalePrice:", np.log1p(train.SalePrice).skew())
plt.hist(np.log1p(train.SalePrice), color='green')
plt.show()
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV
from sklearn import metrics 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from mlxtend.regressor import StackingCVRegressor
X = train.drop(['SalePrice'], axis=1)
y = np.log1p(train['SalePrice'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=2)
categorical_cols = [cname for cname in X.columns if
                    X[cname].nunique() <= 30 and
                    X[cname].dtype == "object"] 
                


numerical_cols = [cname for cname in X.columns if
                 X[cname].dtype in ['int64','float64']]


my_cols = numerical_cols + categorical_cols

X_train = X_train[my_cols].copy()
X_valid = X_valid[my_cols].copy()
X_test = test[my_cols].copy()
num_transformer = Pipeline(steps=[
    ('num_imputer', SimpleImputer(strategy='constant'))
    ])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),       
        ('cat',cat_transformer,categorical_cols),
        ])
def inv_y(transformed_y):
    return np.exp(transformed_y)

n_folds = 10

# XGBoost
model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_valid)
print('XGBoost: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))

      
# Lasso   
model = LassoCV(max_iter=1e6)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_valid)
print('Lasso: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))
  
      
      
# GradientBoosting   
model = GradientBoostingRegressor()
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_valid)
print('Gradient: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))
def inv_y(transformed_y):
    return np.exp(transformed_y)

n_folds = 10

# XGBoost
model = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0,gamma=0, subsample=0.7,colsample_bytree=0.7,objective='reg:squarederror', nthread=-1,scale_pos_weight=1, seed=27, reg_alpha=0.00006)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_valid)
print('XGBoost: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))

      
# Lasso   
model = LassoCV(max_iter=1e7,  random_state=14, cv=n_folds)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_valid)
print('Lasso: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))
  
      
      
# GradientBoosting   
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_valid)
print('Gradient: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_valid))))
n_folds = 10

model = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                     max_depth=3, min_child_weight=0,
                     gamma=0, subsample=0.7,
                     colsample_bytree=0.7,
                     objective='reg:squarederror', nthread=-1,
                     scale_pos_weight=1, seed=27,
                     reg_alpha=0.00006)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])


scores = cross_val_score(clf, X, y, scoring='neg_mean_squared_error', 
                         cv=n_folds)
gbr_mae_scores = -scores

print('RMSE: ' + str(gbr_mae_scores.mean()))
print('Error std deviation: ' +str(gbr_mae_scores.std()))
model = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                     max_depth=3, min_child_weight=0,
                     gamma=0, subsample=0.7,
                     colsample_bytree=0.7,
                     objective='reg:squarederror', nthread=-1,
                     scale_pos_weight=1, seed=27,
                     reg_alpha=0.00006)

final_model = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])

final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': inv_y(final_predictions)})

output.to_csv('submission.csv', index=False)

