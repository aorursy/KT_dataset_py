import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn

import os

from scipy import stats

from scipy.stats import norm, skew

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

from subprocess import check_output

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train.describe()
test.describe()
train_ID = train['Id']

test_ID = test['Id']

#Dropping the 'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
train.info()
test.info()
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'], c='mediumorchid')

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)
#Removing extreme outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Checking the graphic again

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'], c='black')

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)
#Using the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])

#Checking the new distribution 

sns.distplot(train['SalePrice'], color='red', fit=norm);

#Getting the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Plotting the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

#QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

data = pd.concat((train, test)).reset_index(drop=True)
plt.figure(figsize=[40,20])

sns.heatmap(data.corr(), cmap='viridis', annot=True)
fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(14,10))

sns.regplot(x='OverallQual', y = 'SalePrice', color='deepskyblue', data = data, scatter = True,

            fit_reg=True, ax=ax1)

sns.regplot(x='TotalBsmtSF', y = 'SalePrice', color='orchid', data = data, scatter= True,

            fit_reg=True, ax=ax2)

sns.regplot(x='GrLivArea', y = 'SalePrice', color='crimson', data = data, scatter= True,

            fit_reg=True, ax=ax3)

sns.regplot(x='GarageArea', y = 'SalePrice', color='gray', data = data, scatter= True,

            fit_reg=True, ax=ax4)

sns.regplot(x='FullBath', y = 'SalePrice', color='gold', data = data, scatter= True,

            fit_reg=True, ax=ax5)

sns.regplot(x='YearBuilt', y = 'SalePrice', color='yellowgreen', data = data, scatter= True,

            fit_reg=True, ax=ax6)
data_na = (data.isnull().sum() / len(data)) * 100

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :data_na})

missing_data.head(20)
data["PoolQC"] = data["PoolQC"].fillna("None")

data["MiscFeature"] = data["MiscFeature"].fillna("None")

data["Alley"] = data["Alley"].fillna("None")

data["Fence"] = data["Fence"].fillna("None")

data["FireplaceQu"] = data["FireplaceQu"].fillna("None")

data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    data[col] = data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    data[col] = data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    data[col] = data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    data[col] = data[col].fillna('None')

data["MasVnrType"] = data["MasVnrType"].fillna("None")

data["MasVnrArea"] = data["MasVnrArea"].fillna(0)

data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])

data = data.drop(['Utilities'], axis=1)

data["Functional"] = data["Functional"].fillna("Typ")

data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])

data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])

data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])

data['MSSubClass'] = data['MSSubClass'].fillna("None")
#Transforming some features into categorical

data['MSSubClass'] = data['MSSubClass'].apply(str)

data['OverallCond'] = data['OverallCond'].astype(str)

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)
#Transforming features into numeric

from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(data[c].values)) 

    data[c] = lbl.transform(list(data[c].values))        

print('Shape data: {}'.format(data.shape))
numeric_feats = data.dtypes[data.dtypes != "object"].index

#Checking the skew of all numerical features

skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
data.kurt()
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea', 'FullBath',

           'YearBuilt','YearRemodAdd']

sns.pairplot(data[columns], size = 2, kind ='scatter', diag_kind='kde')
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #data[feat] += 1

    data[feat] = boxcox1p(data[feat], lam)

#data[skewed_features] = np.log1p(data[skewed_features])
#Visualizing categorical features

categorical_features = data.select_dtypes(include=[np.object])

def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y, palette='husl')

    x=plt.xticks(rotation=90)

f = pd.melt(data, id_vars=['SalePrice'], value_vars=categorical_features)

g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

g = g.map(boxplot, "value", "SalePrice")
data.drop(['SalePrice'], axis=1, inplace=True)

print("data size is : {}".format(data.shape))
#Check remaining missing values if any 

data_na = (data.isnull().sum() / len(data)) * 100

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :data_na})

missing_data.head()
#Adding total sqfootage feature 

data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
data = pd.get_dummies(data)

print(data.shape)
train = data[:ntrain]

test = data[ntrain:]
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
#Using cross_val_score and shuffling the data

n_folds = 5

def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    #The score which needs to be minimized is negated

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
KRR = KernelRidge(alpha=0.8, kernel='polynomial', degree=2, coef0=3.5)
score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,

                                   max_depth=2, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state=42)
score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.5604, gamma=0.0578, 

                             learning_rate=0.05, max_depth=2, 

                             min_child_weight=1.7817, n_estimators=900,

                             reg_alpha=0.4765, reg_lambda=0.9173,

                             subsample=0.4738, silent=1,

                             random_state =42, nthread = -1)
score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
#Averaging models

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

    #Defining clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        #Training cloned base models

        for model in self.models_:

            model.fit(X, y)

        return self

    #Doing the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#Adding a meta-model and stacking averaged models

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

    #Fitting the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        #Training cloned base models then creating out-of-fold predictions

        #that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

        #Training the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

     #Doing the predictions of all base models on the test data and using the averaged predictions as 

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
#Difining a rmsle evaluation function

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
stacked_averaged_models.fit(train.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(train.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

print(rmsle(y_train, stacked_train_pred))
model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))

print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(train, y_train)

lgb_train_pred = model_lgb.predict(train)

lgb_pred = np.expm1(model_lgb.predict(test.values))

print(rmsle(y_train, lgb_train_pred))
'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')

#Weighted average

print(rmsle(y_train,stacked_train_pred*0.5 +

               xgb_train_pred*0.1 + lgb_train_pred*0.4 ))
ensemble = stacked_pred*0.5 + xgb_pred*0.1 + lgb_pred*0.4
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = ensemble

sub.to_csv('submission.csv',index=False)