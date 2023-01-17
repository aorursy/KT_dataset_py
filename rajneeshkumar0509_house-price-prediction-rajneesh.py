# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

from sklearn.preprocessing import LabelEncoder

from scipy import stats

from scipy.stats import norm, skew

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge

from sklearn import tree

from sklearn import neighbors

from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb

from lightgbm import LGBMRegressor

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
def basic_EDA(df):

    print('------------Top 5 results---------------------')

    print(df.head())

    print('------------Columns of dataset----------------')

    print(df.columns)

    print('------------Dimension of dataset--------------')

    print(df.shape)

    print('------------Data type of dataset--------------')

    print(df.dtypes)

    print('------------Information of dataset------------')

    print(df.info())

    print('------------Statistics of dataset-------------')

    print(df.describe().T)
print('===================Basic EDA of training dataset====================')

basic_EDA(train)
print('===================Basic EDA of test dataset========================')

basic_EDA(test)
#Checking & Plotting missing values of training dataset

missing_count_train = train.isnull().sum()

missing_prcnt_train = (train.isnull().sum()/len(train))*100

missing_train = pd.DataFrame({'missing_count': missing_count_train, 'missing%': missing_prcnt_train}).sort_values(by='missing%', ascending=False)

print(missing_train)



#sns.heatmap(train.isnull(), cmap="viridis", cbar=False, yticklabels=False)

fig, ax = plt.subplots(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=missing_train.index, y='missing%', data=missing_train)

plt.title('Missing% in Training dataset')

plt.show()
#Checking & Plotting missing values of test dataset

missing_count_test = test.isnull().sum()

missing_prcnt_test = (test.isnull().sum()/len(test))*100

missing_test = pd.DataFrame({'missing_count': missing_count_test, 'missing%': missing_prcnt_test}).sort_values(by='missing%', ascending=False)

print(missing_test)



#sns.heatmap(test.isnull(), cmap="viridis", cbar=False, yticklabels=False)

fig, ax = plt.subplots(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=missing_train.index, y='missing%', data=missing_train)

plt.title('Missing% in Test dataset')

plt.show()
# Data imputation



train = train.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1) # Deleting features having >80% missing values

test = test.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1) # Deleting features having >80% missing values

train['FireplaceQu'] = train['FireplaceQu'].fillna('None')

test['FireplaceQu'] = test['FireplaceQu'].fillna('None')

train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].median())

test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].median())

train['GarageYrBlt'].fillna(train['GarageYrBlt'].dropna().mode()[0], inplace=True)

test['GarageYrBlt'].fillna(test['GarageYrBlt'].dropna().mode()[0], inplace=True)

train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].median())

test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].median())



'''for col in ('GarageFinish', 'GarageQual', 'GarageCond', 'GarageType'):

    train[col].fillna(train[col].dropna().mode()[0], inplace=True)

    test[col].fillna(test[col].dropna().mode()[0], inplace=True)

for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond', 'MasVnrType', 'Electrical'):

    train[col].fillna(train[col].dropna().mode()[0], inplace=True)

    test[col].fillna(test[col].dropna().mode()[0], inplace=True)'''



for col in train.columns:

    if train[col].isnull().sum()>0:

        train[col].fillna(train[col].mode()[0], inplace=True)

for col in test.columns:

    if test[col].isnull().sum()>0:

        test[col].fillna(test[col].mode()[0], inplace=True)
temp1 = train.groupby(['MSZoning'])['SalePrice'].max()

temp2 = train.groupby(['MSZoning'])['SalePrice'].mean()



sns.barplot(x=temp1.index, y=temp1.values)

plt.ylabel('Max. SalePrice')

plt.title('Max. SalePrice per MSZoning')

plt.show()



sns.barplot(x=temp2.index, y=temp2.values)

plt.ylabel('Avg. SalePrice')

plt.title('Avg. SalePrice per MSZoning')

plt.show()



colors=['blue','red','yellow','green','brown']

explode=[0,0.2,0,0,0]

plt.pie(temp2.values,explode=explode,labels=temp2.index,colors=colors,autopct='%1.2f%%')

plt.title('SalePrice distribution per MSZoning',color='black',fontsize=10)

plt.show()
plt.figure(figsize=(10,8))

sns.barplot(x=train['MSZoning'], y=train['SalePrice'], hue=train['SaleCondition'])

plt.show()
fig, ax = plt.subplots(figsize=(10,6))

plt.scatter(x=range(train.shape[0]), y=train['SalePrice'], color='y')

plt.title('Sale price distribution')

plt.ylabel('SalePrice')

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'], color='g',alpha=0.9)

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
# Deleting outlier (High living area but very low sale price)

train = train.drop(train[(train['SalePrice']<300000) & (train['GrLivArea']>4000)].index)



# Checking graph again

fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'], color='g',alpha=0.9)

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
fig, ax = plt.subplots(figsize=(10,6))



ax = sns.swarmplot(x = train['HouseStyle'], y = train['SalePrice'], data = train);

sns.set(style ="whitegrid")  

plt.title('SalePrice distribution as per HouseStyle')

plt.show()
fig, ax = plt.subplots(figsize=(10,6))

sns.boxplot(x="MSZoning", y="SalePrice", data=train, palette='rainbow')

plt.title('MSZoning vs SalePrice')

plt.show()
fig, ax = plt.subplots(figsize=(10,6))

sns.boxplot(x=train['GarageType'], y=train['SalePrice'])

plt.title('Garagetype vs SalePrice')

plt.show()
fig, ax = plt.subplots(figsize=(10,6))

fig = sns.boxplot(x=train['OverallQual'], y=train['SalePrice'])

plt.title('OverallQual vs SalePrice')

fig.axis(ymin=0, ymax=800000);

plt.show()
fig, ax = plt.subplots(figsize=(10,6))

sns.boxplot(x=train['OverallCond'], y=train['SalePrice'])

plt.title('OverallCond vs SalePrice')

plt.show()
# Transforming some numerical variables that are really categorical



train['MSSubClass'] = train['MSSubClass'].astype(str)

train['OverallCond'] = train['OverallCond'].astype(str)

train['OverallQual'] = train['OverallQual'].astype(str)

train['YrSold'] = train['YrSold'].astype(str)

train['MoSold'] = train['MoSold'].astype(str)



test['MSSubClass'] = test['MSSubClass'].astype(str)

test['OverallCond'] = test['OverallCond'].astype(str)

test['OverallQual'] = test['OverallQual'].astype(str)

test['YrSold'] = test['YrSold'].astype(str)

test['MoSold'] = test['MoSold'].astype(str)
train.shape
test.shape
# Making copy of Training and Test dataset as a backup

train_copy = train.copy()

test_copy = test.copy()
train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)
train['SalePrice'] = np.log1p(train['SalePrice'])  # Taking log(1+x) transform of target variable

y = train['SalePrice'].reset_index(drop=True)      # Storing target variable as y
train_features = train.drop(['SalePrice'], axis=1)

test_features = test

features = pd.concat([train_features, test_features]).reset_index(drop=True)

features.shape
# Get numerical features and check & transform skew of these features

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

skew_features = features[numerics].apply(lambda x : skew(x)).sort_values(ascending=False)





# Transforming high skew(skew>0.5) features using boxcox1p

for i in skew_features[skew_features > 0.5].index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i]+1))
# One hot encoding of all categorical features

final_features = pd.get_dummies(features).reset_index(drop=True)

final_features.shape
# Splitting final_features dataframe into Training and Submission dataframe

X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(y):, :]

X.shape, y.shape, X_sub.shape
# Defining functions for RMSE of cross-validation and RMSLE (Root Mean Square Logarithmic Error)

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
# Using Linear Regression

lr = LinearRegression()

score = cv_rmse(lr)

lr.fit(X, y)

y_pred_linear = lr.predict(X)



print(' CV Score of Linear Regression model: {:.4f} ({:.4f})'.format(score.mean(), score.std()))

print('RMSLE score on train data:')

print(rmsle(y, y_pred_linear))
# Using Lasso Regression

lasso = Lasso(alpha=0.00001)

score = cv_rmse(lasso)

lasso.fit(X, y)

y_pred_lasso = lasso.predict(X)



print(' CV Score of Lasso Regression model: {:.4f} ({:.4f})'.format(score.mean(), score.std()))

print('RMSLE score on train data:')

print(rmsle(y, y_pred_lasso))
# Using Ridge Regression

ridge = Ridge(alpha=1.0,max_iter=None,tol=0.001,solver='auto',random_state=27)

score = cv_rmse(ridge)

ridge.fit(X, y)

y_pred_ridge = ridge.predict(X)



print(' CV Score of Ridge Regression model: {:.4f} ({:.4f})'.format(score.mean(), score.std()))

print('RMSLE score on train data:')

print(rmsle(y, y_pred_ridge))
# Using Bayesian Ridge Regression

bayridge = BayesianRidge()

score = cv_rmse(bayridge)

bayridge.fit(X, y)

y_pred_bayridge = bayridge.predict(X)



print(' CV Score of Bayesian Ridge Regression model: {:.4f} ({:.4f})'.format(score.mean(), score.std()))

print('RMSLE score on train data:')

print(rmsle(y, y_pred_bayridge))
# Using Decision Tree Regressor

dtr = tree.DecisionTreeRegressor()

score = cv_rmse(dtr)

dtr.fit(X, y)

y_pred_dtr = dtr.predict(X)



print(' CV Score of Decision tree Regression model: {:.4f} ({:.4f})'.format(score.mean(), score.std()))

print('RMSLE score on train data:')

print(rmsle(y, y_pred_dtr))
# Using KNN Regressor

knn = neighbors.KNeighborsRegressor(n_neighbors=5, weights='uniform')

score = cv_rmse(knn)

knn.fit(X, y)

y_pred_knn = knn.predict(X)



print(' CV Score of KNN Regression model: {:.4f} ({:.4f})'.format(score.mean(), score.std()))

print('RMSLE score on train data:')

print(rmsle(y, y_pred_knn))
# Using Gradient Boosting Regressor

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')

score = cv_rmse(gbr)

gbr.fit(X, y)

y_pred_gbr = gbr.predict(X)



print(' CV Score of Gradient Boosting Regression model: {:.4f} ({:.4f})'.format(score.mean(), score.std()))

print('RMSLE score on train data:')

print(rmsle(y, y_pred_gbr))
# Using XG Boost Regressor

xgr = xgb.XGBRegressor()

score = cv_rmse(xgr)

xgr.fit(X, y)

y_pred_xgr = xgr.predict(X)



print(' CV Score of XG Boosting Regression model: {:.4f} ({:.4f})'.format(score.mean(), score.std()))

print('RMSLE score on train data:')

print(rmsle(y, y_pred_xgr))
# Using LGBM Regressor

lgbm = LGBMRegressor(objective='regression', num_leaves=4,learning_rate=0.01, n_estimators=5000, max_bin=200, bagging_fraction=0.75, bagging_freq=5, bagging_seed=7, feature_fraction=0.2, feature_fraction_seed=7, verbose=-1,)

score = cv_rmse(lgbm)

lgbm.fit(X, y)

y_pred_lgbm = lgbm.predict(X)



print(' CV Score of LGBM Regression model: {:.4f} ({:.4f})'.format(score.mean(), score.std()))

print('RMSLE score on train data:')

print(rmsle(y, y_pred_lgbm))
# Submission of prediction



Prediction = np.floor(np.expm1(xgr.predict(X_sub)))

submission = pd.DataFrame({'Id' : test_copy['Id'], 'SalePrice' : Prediction})

submission.to_csv('submission.csv', index=False)