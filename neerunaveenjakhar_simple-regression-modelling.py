import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

train.head()
train.shape
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

print(test.shape)

test.head(5)
#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)



print(train.shape,test.shape)
train.columns
train.describe()
train['SalePrice'].describe()
#histogram and normal probability plot

plt.figure(figsize=(12,8))

sns.distplot(train['SalePrice'], fit=norm);



plt.figure(figsize=(12,8))

res = stats.probplot(train['SalePrice'], plot=plt)
#applying log transformation

target = train['SalePrice']

target = np.log(target)
#transformed histogram and normal probability plot

fig = plt.figure(figsize=(12,8))

sns.distplot(target, fit=norm);

fig = plt.figure(figsize=(12,8))

res = stats.probplot(target, plot=plt)
feature_num = train.select_dtypes(include=[np.number])

feature_num.columns

#sns.heatmap(train.corr(), annot = True)
feature_cat = train.select_dtypes(include=[np.object])

feature_cat.columns
#Finding Correlation coefficients between numeric features and SalePrice

correlation = feature_num.corr()

print(correlation['SalePrice'].sort_values(ascending=False))
sns.set(font_scale=2)

plt.figure(figsize = (50,35))

ax = sns.heatmap(feature_num.corr(), square = True, vmax = .8, annot = True, 

                 annot_kws={"size": 25},fmt='.1f',cmap='PiYG', linewidths=.5)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
k = 10

cols = correlation.nlargest(k, 'SalePrice')['SalePrice'].index

print(cols)

cm = np.corrcoef(train[cols].values.T)

f, ax = plt.subplots(figsize = [14,12])

sns.heatmap(cm, square = True, vmax = 0.8, annot = True, cmap = 'viridis', linewidths=0.01,

            xticklabels = cols.values, yticklabels = cols.values, linecolor = 'white', annot_kws={'size':12})

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
#Here we can see how each feature is correlated with SalePrice.

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show();
sns.scatterplot(x = train['GarageCars'], y = train['SalePrice'])
sns.regplot(x = train['GarageCars'], y = train['SalePrice'], scatter = True, fit_reg = True)
sns.boxplot(train['SalePrice'])
fig, ((ax1,ax2),(ax3,ax4), (ax5,ax6)) = plt.subplots(ncols = 2, nrows = 3, figsize=(12,10))

sns.regplot(x = train['OverallQual'], y = train['SalePrice'], scatter = True, fit_reg = True, ax = ax1)

sns.regplot(x = train['GrLivArea'], y = train['SalePrice'], scatter = True, fit_reg = True, ax = ax2)

sns.regplot(x = train['GarageArea'], y = train['SalePrice'], scatter = True, fit_reg = True, ax = ax3)

sns.regplot(x = train['FullBath'], y = train['SalePrice'], scatter = True, fit_reg = True, ax = ax4)

sns.regplot(x = train['YearBuilt'], y = train['SalePrice'], scatter = True, fit_reg = True, ax = ax5)

sns.regplot(x = train['TotalBsmtSF'], y = train['SalePrice'], scatter = True, fit_reg = True, ax = ax6)
f,ax = plt.subplots(figsize=[12,10])

sns.boxplot(x = train['SaleType'], y = train['SalePrice'])
f,ax = plt.subplots(figsize=[12,10])

fig = sns.boxplot(x = train['OverallQual'], y = train['SalePrice'])
first_quartile = train['SalePrice'].quantile(0.25)

third_quartile = train['SalePrice'].quantile(0.75)

IQR = third_quartile-first_quartile

new_boundary = third_quartile+3*IQR

#train.drop(train[train['SalePrice']>new_boundary].index, axis = 0, inplace = True)
sns.boxplot(train['SalePrice'])
#Concatenate train and test

# train.drop("SalePrice", axis = 1, inplace = True)

total = pd.concat((train, test)).reset_index(drop=True)

print(total.shape, total.columns)

total_df = [train, test]
for dataset in total_df:

    dataset.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
#missing value analysis

missing = total.isnull().sum().sort_values(ascending=False)

missing = missing[missing>0]

missing 
plt.figure(figsize=(15,8))

missing.plot.bar()
feature_tot_num = total.select_dtypes(include=[np.number])

missing_tot_numeric = feature_tot_num.isnull().sum().sort_values(ascending=False)

missing_tot_numeric_percent = (feature_tot_num.isnull().sum()/feature_tot_num.isnull().count()).sort_values(ascending=False)

missing_num_data = pd.concat([missing_tot_numeric, missing_tot_numeric_percent], axis=1,join='outer', keys=['Total Missing Count', '% of Total Observations'])

missing_num_data.index.name =' Numeric Feature'



print(missing_num_data.head(20), feature_tot_num.shape)

print(feature_tot_num['LotFrontage'].isnull().sum())
missing_num_data = missing_num_data[missing_num_data>0]



plt.figure(figsize=(15,8))

missing_num_data.head(10).plot.barh()
feature_tot_cat = total.select_dtypes(include=[np.object])

missing_tot_cat = feature_tot_cat.isnull().sum().sort_values(ascending=False)

missing_tot_cat_percent = (feature_tot_cat.isnull().sum()/feature_tot_cat.isnull().count()).sort_values(ascending=False)

missing_cat_data = pd.concat([missing_tot_cat, missing_tot_cat_percent], axis=1,join='outer', keys=['Total Missing Count', '% of Total Observations'])

missing_cat_data.index.name =' Categorical Feature'

missing_cat_data.head(25)

print(feature_tot_cat.shape)
missing_tot_cat = feature_tot_cat.isnull().sum(axis=0).reset_index()

missing_tot_cat.columns = ['column_name', 'missing_count']

missing_tot_cat = missing_tot_cat.loc[missing_tot_cat['missing_count']>0]

missing_tot_cat = missing_tot_cat.sort_values(by='missing_count')



ind = np.arange(missing_tot_cat.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_tot_cat.missing_count.values, color='b')

ax.set_yticks(ind)

ax.set_yticklabels(missing_tot_cat.column_name.values, rotation='horizontal')

ax.set_xlabel("Missing Observations Count")

ax.set_title("Missing Observations Count - Categorical Features")

plt.show()

for dataset in total_df:

    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())#float

    dataset['BsmtQual'] = dataset['BsmtQual'].fillna(dataset['BsmtQual'].mode()[0])

    dataset['BsmtCond'] = dataset['BsmtCond'].fillna(dataset['BsmtCond'].mode()[0])

    dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna(dataset['BsmtExposure'].mode()[0])

    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna(dataset['BsmtFinType1'].mode()[0])

    dataset['MasVnrType'] = dataset['MasVnrType'].fillna(dataset['MasVnrType'].mode()[0])

    dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean())#float

    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna(dataset['BsmtFinType2'].mode()[0])

    dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].mode()[0])

    dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna(dataset['FireplaceQu'].mode()[0])

    dataset['GarageType'] = dataset['GarageType'].fillna(dataset['GarageType'].mode()[0])

    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].mean())#float

    dataset['GarageFinish'] = dataset['GarageFinish'].fillna(dataset['GarageFinish'].mode()[0])

    dataset['GarageQual'] = dataset['GarageQual'].fillna(dataset['GarageQual'].mode()[0])

    dataset['GarageCond'] = dataset['GarageCond'].fillna(dataset['GarageCond'].mode()[0])

    ########

    dataset['GarageCars'] = dataset['GarageCars'].fillna(dataset['GarageCars'].mean())

    dataset['GarageArea'] = dataset['GarageArea'].fillna(dataset['GarageArea'].mean())

    dataset['SaleType'] = dataset['SaleType'].fillna(dataset['SaleType'].mode()[0])

    dataset['Functional'] = dataset['Functional'].fillna(dataset['Functional'].mode()[0])

    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['KitchenQual'].mode()[0])

    dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(dataset['BsmtHalfBath'].mean())

    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(dataset['BsmtFullBath'].mean())

    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(dataset['TotalBsmtSF'].mean())

    dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(dataset['BsmtUnfSF'].mean())

    dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(dataset['BsmtFinSF2'].mean())

    dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(dataset['BsmtFinSF1'].mean())

    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(dataset['Exterior2nd'].mode()[0])

    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(dataset['Exterior1st'].mode()[0])

    dataset['Utilities'] = dataset['Utilities'].fillna(dataset['Utilities'].mode()[0])

    dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].mode()[0])
#feature_tot_cat = pd.get_dummies(feature_tot_cat)

#feature_tot_cat.shape

#str(feature_tot_cat.isnull().values.sum())
train_df = pd.get_dummies(train, drop_first=True)
df = pd.concat([train, test])

df1 = pd.get_dummies(df, drop_first=True)
train = df1.iloc[: 1460, :]

test = df1.iloc[1460: , :]

print(train.shape,test.shape)
X = train.drop('SalePrice', axis=1)

y = train['SalePrice']

test = test.drop('SalePrice', axis=1)
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer
#split the data to train the model 

X_train,X_test,y_train,y_test = train_test_split(X, target,test_size = 0.3,random_state= 0)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
n_folds = 5

from sklearn.metrics import make_scorer

from sklearn.model_selection import KFold

scorer = make_scorer(mean_squared_error,greater_is_better = False)

def rmse_CV_train(model):

    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)

    rmse = np.sqrt(-cross_val_score(model,X_train,y_train,scoring ="neg_mean_squared_error",cv=kf))

    return (rmse)

def rmse_CV_test(model):

    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)

    rmse = np.sqrt(-cross_val_score(model,X_test,y_test,scoring ="neg_mean_squared_error",cv=kf))

    return (rmse)
lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

train_pre = lr.predict(X_train)

print('rmse on train',rmse_CV_train(lr).mean())

print('rmse on train',rmse_CV_test(lr).mean())
#plot between predicted values and residuals

plt.scatter(train_pre, train_pre - y_train, c = "blue",  label = "Training data")

plt.scatter(y_pred,y_pred - y_test, c = "black",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()
# Plot predictions - Real values

plt.scatter(train_pre, y_train, c = "blue",  label = "Training data")

plt.scatter(y_pred, y_test, c = "black",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
tree_regressor = DecisionTreeRegressor()

tree_regressor.fit(X, y)

tree_regressor.score(X, y)
forest_regressor = RandomForestRegressor()

forest_regressor.fit(X, y)

forest_regressor.score(X, y)
Xgb_regressor = XGBRegressor()

Xgb_regressor.fit(X, y)
y_predict = Xgb_regressor.predict(test)
submission_sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

dataset = pd.DataFrame({

    'Id': submission_sample['Id'],

    'SalePrice': y_predict

})
dataset.to_csv('output.csv', index=False)