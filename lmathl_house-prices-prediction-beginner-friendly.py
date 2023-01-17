import numpy as np 

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

from scipy import stats

from scipy.stats import norm, skew



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# save the Id column for prediction dataframe

test_ID = test['Id']



# drop the Id columns since Id's are not correlated with house prices

train.drop('Id', axis=1, inplace=True)

test.drop('Id', axis=1, inplace=True)
# visualize outliers in a scatter plot

fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
# delete outliers: large GrLivArea but low price

train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)



fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
sns.distplot(train['SalePrice'] , fit=norm);



(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



# plot SalePrice distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



# QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# use np.log1p which applies log(1+x) to all elements of the column

train['SalePrice'] = np.log1p(train['SalePrice'])



# plot the two graphs again

sns.distplot(train['SalePrice'] , fit=norm);



(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution after log-transformation')



fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat([train, test], sort=False).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print('all_data size is : {}'.format(all_data.shape))
# visualize correlation among variables in a heatmap

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.4)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
all_data.drop(['GarageArea'], axis=1, inplace=True)

print('Number of feature columns : {}'.format(len(all_data.columns)))
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index)

print('There are {} columns with missing values: {}'.format(len(all_data_na.index), all_data_na.index.values))
all_data[all_data_na.index].select_dtypes(include=[np.number]).columns.tolist()
# numerical

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF',

            'GarageCars', 'GarageYrBlt', 'MasVnrArea', 'TotalBsmtSF'):

    all_data[col] = all_data[col].fillna(0)



# group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all_data[all_data_na.index].select_dtypes(include=[np.object]).columns.tolist()
# categorical

for col in ('Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'BsmtQual', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual',

       'GarageType', 'MasVnrType', 'MiscFeature', 'PoolQC'):

    all_data[col] = all_data[col].fillna('None')



for col in ('Electrical', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'MSZoning', 'SaleType'):

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    

# All records are "AllPub", except for one "NoSeWa" and 2 NA, Utilities won't help in predictive modelling.

all_data = all_data.drop(['Utilities'], axis=1)

all_data['Functional'] = all_data['Functional'].fillna('Typ')
# check if there are still missing values

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

all_data_na
# MSSubClass: The building class should be categorical

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)



# convert OverallCond to a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)



# Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
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



print('Shape of all_data: {}'.format(all_data.shape))
# add total sqfootage feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# filter numerical features

numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index



# check the skewness of numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness = skewness[abs(skewness) > 0.75]

print('There are {} skewed numerical features to Box Cox transform'.format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    all_data[feat] = boxcox1p(all_data[feat], lam)    
all_data = pd.get_dummies(all_data)

print(all_data.shape)
train = all_data[:ntrain]

test = all_data[ntrain:]
from sklearn.linear_model import ElasticNet, Lasso

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring='neg_mean_squared_error', cv = kf))

    return(rmse)
from sklearn.model_selection import GridSearchCV

lasso = Lasso()

lasso_parameters = {'alpha': [1e-4, 2e-4, 5e-4, 7e-4, 9e-4, 1e-3, 2e-3, 5e-3, 7e-3, 9e-3, 1e-2, 2e-2, 5e-2]}

lasso_regressor = GridSearchCV(lasso, lasso_parameters, scoring='neg_mean_squared_error', cv=5)

lasso_regressor.fit(train, y_train)

print('Best parameters for lasso regression: {}'.format(lasso_regressor.best_params_))
ENet = ElasticNet()

ENet_parameters = {'alpha': [1e-3, 2e-3, 5e-3, 7e-3, 9e3, 1e-2, 2e-2, 5e-2],

                  'l1_ratio': [.1, .2, .3, .4, .5, .6, .7, .8, .9],

                  'random_state': range(0, 10)}

ENet_regressor = GridSearchCV(ENet, ENet_parameters, scoring='neg_mean_squared_error', cv=5)

ENet_regressor.fit(train, y_train)

print('Best parameters for elastic net: {}'.format(ENet_regressor.best_params_))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.001, l1_ratio=.4, random_state=0))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state=7, nthread=-1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin=55, bagging_fraction=0.8,

                              bagging_freq=5, feature_fraction=0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
score = rmsle_cv(lasso)

print('Lasso score: {:.4f} ({:.4f})'.format(score.mean(), score.std()))



score = rmsle_cv(ENet)

print('ElasticNet score: {:.4f} ({:.4f})'.format(score.mean(), score.std()))



score = rmsle_cv(KRR)

print('Kernel Ridge score: {:.4f} ({:.4f})'.format(score.mean(), score.std()))



score = rmsle_cv(GBoost)

print('Gradient Boosting score: {:.4f} ({:.4f})'.format(score.mean(), score.std()))



score = rmsle_cv(model_xgb)

print('Xgboost score: {:.4f} ({:.4f})'.format(score.mean(), score.std()))



score = rmsle_cv(model_lgb)

print('LGBM score: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                # train cloned base models

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                # predict the untouched holdout with the trained first stage model

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # train the cloned meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    # do the predictions of all base models on the test data and use the averaged predictions as 

    # meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, model_xgb),

                                                 meta_model = lasso)



score = rmsle_cv(stacked_averaged_models)

print('Stacking Averaged models score: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
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



print('RMSLE score on train data:')

print(rmsle(y_train, stacked_train_pred*0.70 + xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = ensemble

sub.to_csv('house_prices_predictions.csv',index=False)