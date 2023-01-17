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
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



df_train.head()
df_train.describe()
df_train["YrSold"].value_counts()


%matplotlib inline

import matplotlib.pyplot as plt  

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')





df_train["SalePrice"].hist(bins = 100, figsize = (10,5))

plt.show()
Id_train = df_train["Id"]

Id_test = df_test["Id"]



df_train.drop("Id", axis = 1, inplace = True)

df_test.drop("Id", axis =1, inplace = True)

df_train.head()
fig, ax = plt.subplots()

ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.show()
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)
fig, ax = plt.subplots()

ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'])

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.show()
from scipy import stats

from scipy.stats import norm, skew

sns.distplot(df_train['SalePrice'] , fit=norm)



mu, sigma = norm.fit(df_train["SalePrice"] )



print("mu = {:.2f} , sigma = {:.2f}".format(mu,sigma))

#QQ plot

fig = plt.figure()

res = stats.probplot(df_train["SalePrice"], plot= plt)

plt.show()
df_train["SalePrice"] = np.log(df_train["SalePrice"])



sns.distplot(df_train['SalePrice'] , fit=norm)



mu, sigma = norm.fit(df_train["SalePrice"] )



print("mu = {:.2f} , sigma = {:.2f}".format(mu,sigma))
#QQ plot

fig = plt.figure()

res = stats.probplot(df_train["SalePrice"], plot= plt)

plt.show()

n_train = df_train.shape[0]

n_test = df_test.shape[0]

y_train = df_train["SalePrice"]

df = pd.concat((df_train, df_test)).reset_index(drop = True)

df.drop(["SalePrice"], axis = 1, inplace = True)

df
missing_vl_ratio = df.isna().mean().round(4) * 100

missing_vl_ratio = missing_vl_ratio.drop(missing_vl_ratio[missing_vl_ratio == 0].index).sort_values(ascending=False)[:30]

dataframe_miss=pd.DataFrame(missing_vl_ratio, columns=['missing ratio'])

dataframe_miss.head(80)
for item in ('PoolQC','MiscFeature', 'Alley', 'Fence', 'FireplaceQu'):

    df[item] = df[item].fillna('None')
df["LotFrontage"] = df.groupby("Neighborhood").transform(lambda x: x.fillna(x.median()))
for item in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure','MSSubClass', 'BsmtFinType1', 'BsmtFinType2'):

    df[item] = df[item].fillna('None')
for item in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df[item] = df[item].fillna(0)
df["MasVnrType"] = df["MasVnrType"].fillna("None")

df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])

df["Utilities"].value_counts()
df = df.drop(['Utilities'], axis=1)
df["Functional"].value_counts()
df["Functional"] = df["Functional"].fillna("Typ") #Check data disc !

print(df["Electrical"].mode())

df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0]) #most common 
print(df["KitchenQual"].mode())

df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0]) #most common 

df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])

df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])

df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
missing_vl_ratio = df.isna().mean().round(4) * 100

missing_vl_ratio = missing_vl_ratio.drop(missing_vl_ratio[missing_vl_ratio == 0].index).sort_values(ascending=False)[:30]

dataframe_miss=pd.DataFrame(missing_vl_ratio, columns=['missing ratio'])

dataframe_miss.head(80)
print(df["MSSubClass"].value_counts())

df["MSSubClass"] = df["MSSubClass"].apply(str)



print(df["OverallCond"].value_counts())

df["OverallCond"] = df["OverallCond"].astype(str)

df["YrSold"].value_counts()

df["YrSold"] = df["YrSold"].astype(str)

df["MoSold"] = df["MoSold"].astype(str)



from sklearn.preprocessing import LabelEncoder

items = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')





for item in items:

    labelencoder = LabelEncoder()

    df[item] = labelencoder.fit_transform(df[item].values)

 

df
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

from scipy.special import boxcox1p



numeric_feats = df.dtypes[df.dtypes != "object"].index

skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewed_features = skewness.index

lam = 0.15

for feature in skewed_features:

    df[feature] = boxcox1p(df[feature], lam)
df = pd.get_dummies(df)

df



df_train = df[:n_train]

df_test = df[n_train:]
df
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
n_folds = 5 # number of folds

def get_cv_scores(model, X, y, print_scores=True):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X) # create folds

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf)) # get rmse

    if print_scores:

        print(f'Root mean squared error: {rmse.mean():.3f} ({rmse.std():.3f})')

    return [rmse]
lasso_model = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

rf = RandomForestRegressor()

gbr = GradientBoostingRegressor()

xgb_model = xgb.XGBRegressor()

lgb_model = lgb.LGBMRegressor()

for model in [lasso_model, rf, gbr, xgb_model, lgb_model]:

    print(str(model))

    get_cv_scores(model, df_train, y_train)
class WeightedAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models, weights):

        self.models = models

        self.weights = weights

        assert sum(self.weights)==1

        

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)

        return self

    

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.sum(predictions*self.weights, axis=1)
weighted_average_model = WeightedAveragedModels([gbr, lasso_model, xgb_model], [0.3, 0.45, 0.25])

get_cv_scores(weighted_average_model, df_train, y_train);

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
weighted_average_model.fit(df_train.values, y_train)

weighted_train_pred = weighted_average_model.predict(df_train.values)

weighted_pred = np.expm1(weighted_average_model.predict(df_test.values))

print(rmsle(y_train, weighted_train_pred))
sub = pd.DataFrame()

sub['Id'] = Id_test

sub['SalePrice'] = weighted_pred

sub.to_csv('submission.csv',index=False)