# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train_size = train.shape[0]
train.shape
# drop Id column since it is useless for modeling/prediction purpose
train.drop('Id', axis=1, inplace=True)
train.shape
test.head()
test.shape
test_id = test['Id']
test.drop('Id', axis=1, inplace=True)
test.shape
# check if the test data set have the same column names as the training data set
misalign_cols = []
for i in range(len(test.columns)):
    if train.columns[i]!=test.columns[i]:
        misalign_cols.append((train.columns[i],test.columns[i]))
len(misalign_cols)
target = 'SalePrice'
train[target].describe()
# plot the distribution of the target
sns.distplot(train[target],fit=norm)
(mu, sigma) = norm.fit(train[target])
plt.legend(['Normal distribution with $\mu=$ {:.0f} and $\sigma=$ {:.0f}'.format(mu,sigma)],loc='best')
plt.ylabel('Probability')
plt.title('Distribution of ' + target)

# plot the QQ plot
fig = plt.figure()
res = stats.probplot(train[target], plot=plt)
plt.show()
# check skewness and kurtosis
print("Skewness: %f" % train[target].skew())
print("Kurtosis: %f" % train[target].kurt())
# add a new variable which is the logarithm transformation of the target
# There are two reasons to do so:
# (1) normal distribution is an assumption for linear regression
# (2) to avoid overflow
train['log_' + target] = np.log1p(train[target])
target = 'log_' + target
# plot the distribution of the new target
sns.distplot(train[target],fit=norm)
(mu, sigma) = norm.fit(train[target])
plt.legend(['Normal distribution with $\mu=$ {:.0f} and $\sigma=$ {:.0f}'.format(mu,sigma)],loc='best')
plt.ylabel('Probability')
plt.title('Distribution of ' + target)

# plot the QQ plot
fig = plt.figure()
res = stats.probplot(train[target], plot=plt)
plt.show()
# scatter plot between target and 'GrLivArea' after log transformation
plt.scatter(train['GrLivArea'], train['SalePrice'])
train.loc[(train['GrLivArea']>4000),['GrLivArea','SaleCondition']]
test.loc[(test['GrLivArea']>4000),['GrLivArea','SaleCondition']]
# concatenate train and test data sets to keep consistent between them
all_data = pd.concat([train, test]).reset_index(drop=True)
# drop features not in the test data set
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.drop(['log_SalePrice'], axis=1, inplace=True)
all_data.shape
# find columns with missing data
cols_with_missing = [col for col in all_data.columns if all_data[col].isnull().any()]

# list the number/percentage of missing data for each column
number_of_missing_data = all_data.isnull().sum().sort_values(ascending=False)
percent_of_missing_data = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([number_of_missing_data, 
                          percent_of_missing_data], 
                          axis=1, 
                          keys=['Total','Percent'])
missing_data.head(len(cols_with_missing))
# replace missing data with None based on data descripton
features = ['PoolQC',
            'MiscFeature',
            'Alley',
            'Fence',
            'FireplaceQu',
            'GarageCond',
            'GarageQual',
            'GarageFinish',
            'GarageType',
            'BsmtFinType1',
            'BsmtFinType2',
            'BsmtExposure',
            'BsmtCond',
            'BsmtQual',
            'MasVnrType']
for feature in features:
    all_data[feature].fillna('None',inplace=True)
# replace missing value with 0 for MasVnrArea
all_data['MasVnrArea'].fillna(0,inplace=True)
# replace missing value with typ for Functional based on data description
all_data['Functional'].fillna('Typ',inplace=True)
# check missing values for other basement related features
all_data.loc[all_data['BsmtHalfBath'].isnull(),
             ['BsmtHalfBath',
              'BsmtFullBath',
              'BsmtFinSF1',
              'BsmtFinSF2',
              'BsmtUnfSF',
              'TotalBsmtSF',
              'BsmtCond']]
# For other basement related variables,
# replace missing value with None for categorical variables, and
# replace missing value with 0 for numerical variables.
features = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtHalfBath','BsmtFullBath']
for feature in features:
    all_data[feature].fillna(0,inplace=True)
# check missing values for other garage related features
all_data.loc[all_data['GarageCars'].isnull(),['GarageCars',
                                              'GarageArea',
                                              'GarageYrBlt',
                                              'GarageCond']]
# replace missing value with 0 for other garage related variables
features = ['GarageCars','GarageArea','GarageYrBlt']
for feature in features:
    all_data[feature].fillna(0,inplace=True)
features = ['Exterior2nd','Exterior1st','SaleType','KitchenQual','Electrical']
for feature in features:
    print(all_data.loc[all_data[feature].isnull(),features])
# replace missing value with most frequent value in training data set
features = ['Exterior2nd','Exterior1st','SaleType','KitchenQual','Electrical']
for feature in features:
    all_data[feature].fillna(all_data[feature].mode()[0], inplace=True)
all_data['Utilities'].value_counts()
# delete Utilities since it is not helpful for prediction
all_data.drop(['Utilities'], axis=1, inplace=True)
# replace missing value with the median one in the same neighborhood for LotFrontage
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# replace missing value with the most frequent one in the same neighborhood for MSZoning
all_data['MSZoning'] = all_data.groupby('Neighborhood')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# note that ideally, we should not use the information in the test data, but only the training data, to impute this missing number
# find columns with missing data
cols_with_missing = [col for col in all_data.columns if all_data[col].isnull().any()]

# list the number/percentage of missing data for each column
number_of_missing_data = all_data.isnull().sum().sort_values(ascending=False)
percent_of_missing_data = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([number_of_missing_data, 
                          percent_of_missing_data], 
                          axis=1, 
                          keys=['Total','Percent'])
missing_data.head(len(cols_with_missing))
# check numerical features
num_features = [col for col in all_data.columns if all_data[col].dtype 
                    in [np.int64,np.float64]]
print(len(num_features))
print(num_features)
# convert MSSubClass to categorical since there is no ordering in this variable
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
# check numerical features
num_features = [col for col in all_data.columns if all_data[col].dtype 
                    in [np.int64,np.float64]]
print(len(num_features))
print(num_features)
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
# check skewness of numerical features
skewed_features = all_data[num_features].apply(lambda x: x.dropna().skew()).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed_features})
skewness.head(35)
# apply box cox transformation
features = skewness[abs(skewness.Skew)>1].index
for feature in features:
    all_data[feature] = np.log1p(all_data[feature])
# check categorical features
cat_features = [col for col in all_data.columns if all_data[col].dtypes == 'object']
print(len(cat_features))
print(cat_features)
all_data = pd.get_dummies(all_data)
print(all_data.shape)
y_train = train['log_SalePrice'].values
train = all_data[:train_size]
test = all_data[train_size:]
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LassoLarsCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

random_state = 17
rbscaler = RobustScaler().fit(train.values)
stdscaler = StandardScaler().fit(train.values)
#Validation function
n_folds = 5

def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=17).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
# I found the following hyperparameter by cross validation and grid search
ridge = make_pipeline(RobustScaler(), Ridge(alpha=5, random_state=random_state))
score = rmse_cv(ridge)
print("Ridge regression with robust scaler:")
print(score.mean())
print(score.std())
ridge = Ridge(alpha=5,random_state=random_state)
ridge_transformer = RobustScaler().fit(train.values)
ridge.fit(ridge_transformer.transform(train.values), y_train)
print(np.sqrt(mean_squared_error(ridge.predict(ridge_transformer.transform(train.values)),y_train)))

ridge_pred = np.expm1(ridge.predict(ridge_transformer.transform(test.values)))
result = pd.DataFrame({'Id': test_id, 'SalePrice': ridge_pred})
result.to_csv('submission.csv',index=False)
# I found the following hyperparameter by cross validation and grid search.
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=random_state))
score = rmse_cv(lasso)
print("Ridge regression with robust scaler:")
print(score.mean())
print(score.std())
lasso = Lasso(alpha=0.0005,random_state=random_state)
lasso_transformer = RobustScaler().fit(train.values)
lasso.fit(lasso_transformer.transform(train.values), y_train)
print(np.sqrt(mean_squared_error(lasso.predict(lasso_transformer.transform(train.values)),y_train)))

lasso_pred = np.expm1(lasso.predict(lasso_transformer.transform(test.values)))
result = pd.DataFrame({'Id': test_id, 'SalePrice': lasso_pred})
result.to_csv('submission.csv',index=False)
# I found the following hyperparameters by cross validation and grid search
enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0008, l1_ratio=0.4, random_state=random_state))
score = rmse_cv(enet)
print("Enet regression with robust scaler:")
print(score.mean())
print(score.std())
enet.fit(rbscaler.transform(train.values), y_train)
print(np.sqrt(mean_squared_error(enet.predict(rbscaler.transform(train.values)),y_train)))

enet_pred = np.expm1(enet.predict(rbscaler.transform(test.values)))
result = pd.DataFrame({'Id': test_id, 'SalePrice': enet_pred})
result.to_csv('submission.csv',index=False)
# I found the following parameters by grid search
gbm = GradientBoostingRegressor(n_estimators=1500, 
                                learning_rate=0.01,
                                max_depth=3, 
                                max_features='sqrt',
                                min_samples_leaf=3,  
                                loss='huber', 
                                random_state=random_state)
score = rmse_cv(gbm)
print(score.mean())
print(score.std())
gbm.fit(train.values, y_train)
print(np.sqrt(mean_squared_error(gbm.predict(train.values), y_train)))

gbm_pred = np.expm1(gbm.predict(test.values))
result = pd.DataFrame({'Id': test_id, 'SalePrice': gbm_pred})
result.to_csv('submission.csv',index=False)
import xgboost as xgb

# I found the following parameters by cross validation, grid/random search
xgb = xgb.XGBRegressor(colsample_bytree=0.680455368796, 
                               gamma=0.0218720959, 
                               learning_rate=0.01, 
                               max_depth=3, 
                               min_child_weight=1.15901838, 
                               n_estimators=2900,
                               reg_alpha=0.475770252, 
                               reg_lambda=0.88246737752,
                               subsample=0.515, 
                               silent=1,
                               random_state=random_state, 
                               nthread=-1)

score = rmse_cv(xgb)
print(score.mean())
print(score.std())

xgb.fit(train.values, y_train)
print(np.sqrt(mean_squared_error(xgb.predict(train.values),y_train)))

xgb_pred = np.expm1(xgb.predict(test.values))
result = pd.DataFrame({'Id': test_id, 'SalePrice': xgb_pred})
result.to_csv('submission.csv',index=False)
import lightgbm as lgb

# I found the following parameters by cross validation, grid/random search
lgb = lgb.LGBMRegressor(objective='regression',
                            num_leaves=4,
                            min_data_in_leaf=1,
                            bagging_fraction=0.727,
                            bagging_freq=5,
                            bagging_seed=17,
                            feature_fraction=0.5,
                            feature_fraction_seed=19,
                            max_bin=30, 
                            learning_rate=0.01, 
                            n_estimators=6300,
                            min_sum_hessian_in_leaf=6)
score = rmse_cv(lgb)
print(score.mean())
print(score.std())

lgb.fit(train.values, y_train)
print(np.sqrt(mean_squared_error(lgb.predict(train.values),y_train)))

lgb_pred = np.expm1(lgb.predict(test.values))
result = pd.DataFrame({'Id': test_id, 'SalePrice': lgb_pred})
result.to_csv('submission.csv',index=False)
# borrow the idea from random forest, just average
# I also notied someone borrowed the idea from boosting, i.e., fit another model to the residual
pred = (ridge_pred + lasso_pred + enet_pred + xgb_pred + lgb_pred)/5
result = pd.DataFrame({'Id': test_id, 'SalePrice': pred})
result.to_csv('submission.csv',index=False)