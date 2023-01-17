# Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import missingno as msno

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from matplotlib.gridspec import GridSpec

from collections import Counter

from scipy.stats import norm

from scipy import stats

import warnings



warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)

pd.set_option('display.width', 200)

pd.set_option('display.float_format', lambda x: f'{x: .3f}')

%matplotlib inline
# load files

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



all_data = pd.concat([train, test]).reset_index(drop=True).drop('SalePrice', axis=1)

y_train = train['SalePrice']

test_ID = test['Id']
# Display the first 5 rows of the train dataset.

train.head()
# Display the first 5 rows of the train dataset.

test.head()
# Data Overview

print('train data: %d rows, %d columns.' % train.shape)

print('test data: %d rows, %d columns.' % test.shape)

print('Overview'.center(50, '-'))

print(train.info())

# Index column: Id, target column: SalePrice, 36 numeric columns, 43 categorical columns.
# Missing values visualization(all data)

msno.matrix(train, labels=True)

missing_count = all_data.isnull().sum().sort_values(ascending=False)

missing_rate = all_data.isnull().mean().sort_values(ascending=False)

missing_data = pd.concat([missing_count, missing_rate], axis=1, keys=['missing_count', 'missing_rate'])

missing_data = missing_data[missing_data['missing_rate'] > 0]

print('Missing Rate'.center(50, '-'))

print(missing_data)



plt.figure(figsize=(20, 10))

sns.barplot(x=missing_data.index, y=missing_data['missing_rate'] * 100)

plt.xticks(rotation=45)

plt.yticks(np.arange(0, 110, 10))

plt.title('Missing Rate', fontsize=18)

print('missing columns:', missing_data.sort_index().index)
# Imputing missing values



# refer to data description

all_data['Alley'] = all_data['Alley'].fillna('None')

all_data['PoolQC'] = all_data['PoolQC'].fillna('None')

all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')

all_data['Fence'] = all_data['Fence'].fillna('None')

all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')



# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



for col in ['GarageQual', 'GarageFinish', 'GarageCond', 'GarageType']:

    all_data[col] = all_data[col].fillna('None')

for col in ['GarageArea', 'GarageCars', 'GarageYrBlt']:

    all_data[col] = all_data[col].fillna(0)



# Basement

for col in ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']:

    all_data[col] = all_data[col].fillna('None')

for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF']:

    all_data[col] = all_data[col].fillna(0)



all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])



all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)

all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')



all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0])



all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)

all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(all_data['TotalBsmtSF'].mode()[0])



print(all_data.isnull().sum().max())
# Transforming some numerical variables that are really categorical

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
# Adding total sqfootage feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# Label Encoding some categorical variables that may contain information in their ordering set

from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))
# Numeric data

numeric_columns = all_data.select_dtypes(include='number').columns[1:]  # Drop Id

numeric_data = all_data[numeric_columns]

num_des = numeric_data.describe().T.assign(nunique=numeric_data.apply(lambda x: x.nunique()),

                         missing_rate=numeric_data.apply(lambda x: x.isnull().mean()),

                         mode=numeric_data.apply(lambda x: x.mode()[0]),

                         mode_pct=numeric_data.apply(lambda x: sum(x == x.mode()[0]) / len(train)),

                         skew=numeric_data.apply(lambda x: x.dropna().skew())).sort_values('mode_pct', ascending=False)

num_des

# In numeric data, the highest missing rate is 17.7% which column is LotFrontage, we will keep it.

# KitchenAbvGr,3SsnPorch,ScreenPorch,PoolArea,MiscVal,BsmtHalfBath,LowQualFinSF:mode_pct >=0.9,we will drop them.
# Box Cox Transformation of (highly) skewed features

skewness = num_des[abs(num_des['skew']) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam)

    

#all_data[skewed_features] = np.log1p(all_data[skewed_features])
# Categorical data

categorical_columns = train.select_dtypes(include='object').columns

train[categorical_columns].describe().T.assign(top_pct=lambda x: x['freq'] / len(train),

                                    missing_rate=train.apply(lambda x: x.isnull().mean())).sort_values('top_pct', ascending=False)

# In categorical data, there are 11 columns which top_pct >= 0.9: 

# Street,Utilities,LandSlope,Condition2,RoofMatl,Heating,CentralAir,Electrical,Functional,GarageCond,PavedDrive, we will drop them later.

# There are 3 columns which missing rate >= 0.9:Alley,PoolQC,MiscFeature, we will drop them later.
# Getting dummy categorical features

dummy_data = pd.get_dummies(all_data[categorical_columns])

all_data.drop(categorical_columns, axis=1, inplace=True)

all_data = pd.concat([all_data, dummy_data], axis=1)



train = all_data[:len(train)]

test = all_data[len(train):]

print(all_data.shape)
train.sample(5)
# SalePrice

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.distplot(train['SalePrice'], ax=ax1)

sns.boxplot(y=train['SalePrice'], ax=ax2)

plt.title('SalePrice(mean=%d)' % int(train['SalePrice'].mean()))

print('skewness:', train['SalePrice'].skew())

print('kurtosis:', train['SalePrice'].kurtosis())

train['SalePrice'].describe()
# Fill in missing values of categoric data.

train[categorical_columns] = train[categorical_columns].fillna('missing')

print('Max missing rate in categoric data: %.2f%%' % (train[categorical_columns].isnull().sum().max() / len(train)*100))
# Numeric data visualization

grid = GridSpec(37, 3)

plt.figure(figsize=(10 * 3, 37 * 6))

for i, col in enumerate(numeric_columns):

    # distplot

    ax1 = plt.subplot(grid[i * 3])

    sns.distplot(train[col].fillna(-1), color='g', ax=ax1)

    if col != 'SalePrice':

        sns.distplot(test[col].fillna(-1), color='r', ax=ax1)

    ax1.set_title(f'{col} distribution', fontsize=18)

    ax1.set_ylabel('Density', fontsize=15)

    ax1.set_xlabel(None)

    

    # scatter plot

    ax2 = plt.subplot(grid[i * 3 + 1])

    ax2.scatter(x=col, y='SalePrice', data=train, color='g')

    ax2.set_title(col, fontsize=18)

    

    # boxplot

    ax3 = plt.subplot(grid[i * 3 + 2])

    sns.boxplot(x=col, data=train, ax=ax3)

    ax3.set_title(col, fontsize=18)

    ax3.set_xlabel(None)



plt.show()
numeric_cols_box = train.select_dtypes(include='number').nunique()

numeric_cols_box = numeric_cols_box[numeric_cols_box <= 20]

nrows = int(np.ceil(len(numeric_cols_box) / 4))

gs = GridSpec(nrows, 4)

plt.figure(figsize=(20, nrows*6))

for i, col in enumerate(numeric_cols_box.index.values):

    ax = plt.subplot(gs[i])

    sns.boxplot(x=col, y='SalePrice', data=train, ax=ax)
var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(25, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90)
# Correlations between numeric columns

corrmat = train.drop('Id', axis=1).select_dtypes(include='number').corr()

plt.figure(figsize=(20, 18))

sns.heatmap(corrmat, cmap=plt.cm.Greens, fmt='.1f', annot=True, linecolor='w', linewidths=.1)
# Plot heatmap between SalePrice and other columns which has the highest top 10 corr.

plt.figure(figsize=(12, 9))

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values, cmap=plt.cm.Greens)

plt.show()
# Scatter plots between 'SalePrice' and correlated variables

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)
# Categorical data visualization

dtypes = train.dtypes

categorical_columns = dtypes[dtypes == 'object'].index

grid = GridSpec(43, 2)

plt.figure(figsize=(10 * 2, 43 * 6))



for i, col in enumerate(categorical_columns):

    # bar

    train_values = train[col].fillna('missing')

    test_values = test[col].fillna('missing')

    labels = set(train_values.unique().tolist() + test_values.unique().tolist())



    train_cnts = [train_values.value_counts().to_dict().get(label, 0) for label in labels]

    test_cnts = [test_values.value_counts().to_dict().get(label, 0) for label in labels]

   

    x_range = np.arange(len(labels))

    width = 0.35

    try:

        ax1 = plt.subplot(grid[2 * i])

    except:

        print(i)

    ax1.bar(x_range - width / 2, train_cnts, width, label='train set')

    ax1.bar(x_range + width / 2, test_cnts, width, label='test set')

    ax1.set_title(col + ' bar', fontsize=18)

    ax1.set_xticks(x_range)

    ax1.set_xticklabels(labels, rotation=45)

    plt.legend()

    

    # boxplot

    ax2 = plt.subplot(grid[2 * i + 1])

    sns.boxplot(x=col, y='SalePrice', data=train, ax=ax2)

    ax2.set_title(col, fontsize=18)

    ax2.set_xlabel('')

plt.show()
# Outliers of SalePrice

#standardizing data

saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
# Bivariate analysis

#deleting outlier points

train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice']<300000)].index)

train = train.drop(train[train['LotFrontage'] > 300].index)

train = train.drop(train[train['LotArea'] > 100000].index)

train = train.drop(train[train['MasVnrArea'] > 1450].index)

train = train.drop(train[train['BsmtFinSF1'] > 5000].index)

train = train.drop(train[train['TotalBsmtSF'] > 5000].index)

train = train.drop(train[train['1stFlrSF'] > 4000].index)

train = train.drop([581, 1190, 1061])
# Normality Homoscedasticity Linearity Absence of correlated errors

# In the search for normality

#histogram and normal probability plot

(mu, sigma) = norm.fit(y_train)

plt.title('Norm dist: u={:.2f},sigma={:.2f}'.format(mu, sigma))

sns.distplot(y_train, fit=norm);

fig = plt.figure()

res = stats.probplot(y_train, plot=plt)
# Ok, 'SalePrice' is not normal. It shows 'peakedness', positive skewness and does not follow the diagonal line.

#But everything's not lost. A simple data transformation can solve the problem. 

#This is one of the awesome things you can learn in statistical books: in case of positive skewness,

#log transformations usually works well. When I discovered this, I felt like an Hogwarts' student discovering a new cool spell.

#applying log transformation

y_train = np.log(y_train)

#transformed histogram and normal probability plot

sns.distplot(y_train, fit=norm);

fig = plt.figure()

res = stats.probplot(y_train, plot=plt)
#histogram and normal probability plot

sns.distplot(train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot=plt)
#data transformation

train['GrLivArea'] = np.log(train['GrLivArea'])

#transformed histogram and normal probability plot

sns.distplot(train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['TotalBsmtSF'], plot=plt)
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

def log_transform(train):

    train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)

    train['HasBsmt'] = 0 

    train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1



    #transform data

    train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

    return train
train = log_transform(train)

#histogram and normal probability plot

sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
# In the search for writing 'homoscedasticity' right at the first attempt

#scatter plot

plt.scatter(train['GrLivArea'], train['SalePrice']);
#scatter plot

plt.scatter(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], train[train['TotalBsmtSF']>0]['SalePrice']);
# Delete columns

train = train.drop(['TotRmsAbvGrd', 'EnclosedPorch', 'OpenPorchSF', 'HalfBath', 'WoodDeckSF', 'YrSold', 'YearRemodAdd', 'MoSold',

                    'OverallCond', 'BsmtFinSF1', 'BsmtFinSF2', 'OverallQual', 'GarageCars', 'MasVnrArea', '2ndFlrSF', 'BedroomAbvGr',

                    'Fireplaces', 'LotFrontage', 'BsmtFullBath', 'FullBath', 'LotArea', 'MSSubClass'], axis=1)

train = train.drop(['KitchenAbvGr','3SsnPorch','ScreenPorch','PoolArea','MiscVal','BsmtHalfBath','LowQualFinSF'], axis=1)

train = train.drop(['Street','Utilities','LandSlope','Condition2','RoofMatl','Heating','CentralAir','Electrical','Functional',

                    'GarageCond','PavedDrive'], axis=1)

train = train.drop(['Alley','PoolQC','MiscFeature'], axis=1)
# Fill in missing values of numeric data.

train = train[train.select_dtypes(include='number').columns]

train = train.apply(lambda x: x.fillna(x.mean()), axis=1)

print('Max missing rate:%.2f%%' % (train.isnull().sum().max() / len(train) * 100))
# Fit linearRegression Model

X_train = train.drop(['Id', 'SalePrice'], axis=1)

y_train = train['SalePrice']

X_train_scale = StandardScaler().fit_transform(X_train)
# prediction visualization

def plot_mse(y_pred, title, response=False):

    plt.figure(figsize=(15, 6))

    x = np.arange(len(y_pred))

    if str(response) != 'False':

        sort = np.argsort(response)

        plt.plot(x, y_pred[sort], color='r', label='Pred')

        plt.plot(x, np.sort(response), color='g', label='Real')

        plt.title(f'{title}(MSE={round(mean_squared_error(response, y_pred))})', fontsize=18)

    else:

        plt.plot(x, np.sort(y_pred), color='r', label='Pred')

    plt.legend()

    plt.show()
# Models

lr = LinearRegression()

RF = RandomForestRegressor(n_estimators=200, n_jobs=-1)

GBDT = GradientBoostingRegressor()

lr.fit(X_train_scale, y_train)

RF.fit(X_train_scale, y_train)

GBDT.fit(X_train_scale, y_train)

y_pred_lr = lr.predict(X_train_scale)

y_pred_RF = RF.predict(X_train_scale)

y_pred_GBDT = GBDT.predict(X_train_scale)

plot_mse(y_pred_lr, title='Linear Regression', response=y_train)

plot_mse(y_pred_RF, title='RandomForestRegressor', response=y_train)

plot_mse(y_pred_GBDT, title='GradientBoostingRegressor', response=y_train)
# Metrics on test set

y_test = np.log(test['SalePrice'])

num_cols = [col for col in numeric_columns if col != 'SalePrice']

test[num_cols] = test[num_cols].apply(lambda x: x.fillna(x.mean()), axis=1)

test[categorical_columns] = test[categorical_columns].fillna('missing')

test = dummy_encode(test)

test['GrLivArea'] = np.log(test['GrLivArea'])

test = log_transform(test)

test = test[X_train.columns].reindex(columns=X_train.columns)



y_pred_test_lr = lr.predict(test)

y_pred_test_RF = RF.predict(test)

y_pred_test_GBDT = GBDT.predict(test)



plot_mse(y_pred_test_lr, title='Linear Regression', response=y_test)

plot_mse(y_pred_test_RF, title='RandomForestRegressor', response=y_test)

plot_mse(y_pred_test_GBDT, title='GradientBoostingRegressor', response=y_test)
# Existing problem: overfit
# Stacking models

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
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train.values, scoring="neg_mean_squared_error", cv = kf))

    return rmse
# Base Models
# LASSO Regression

# This model may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's Robustscaler() method on pipeline

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
# Elastic Net Regression :

# again made robust to outliers

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
# Kernel Ridge Regression

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
# Gradient Boosting Regression :

# With huber loss that makes it robust to outliers

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
# XGBoost :

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
# LightGBM

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
# Base models scores

# Let's see how these base models perform on the data by evaluating the cross-validation rmsle error
score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

# Stacking models

# Simplest Stacking approach : Averaging base models

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)  
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
# Ensembling StackedRegressor, XGBoost and LightGBM

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
# StackedRegressor

stacked_averaged_models.fit(train.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(train.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

print(rmsle(y_train, stacked_train_pred))
# XGBoost

model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))

print(rmsle(y_train, xgb_train_pred))
# LightGBM

model_lgb.fit(train, y_train)

lgb_train_pred = model_lgb.predict(train)

lgb_pred = np.expm1(model_lgb.predict(test.values))

print(rmsle(y_train, lgb_train_pred))
'''RMSE on the entire Train data when averaging'''



print('RMSLE score on train data:')

print(rmsle(y_train,stacked_train_pred*0.70 +

               xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
# Ensemble prediction

ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
# Submission

sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = ensemble

sub.to_csv('submission.csv',index=False)