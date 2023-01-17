# Import useful libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from scipy import stats

from scipy.stats import norm, skew, boxcox_normmax

from scipy.special import boxcox1p

import warnings

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, Ridge

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn_pandas import DataFrameMapper

import xgboost as xgb

import lightgbm as lgb

warnings.filterwarnings('ignore')

%matplotlib inline

#bring in the six packs

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.columns
train.shape
test.shape
#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



train = train.drop('Id', axis=1)

test = test.drop('Id', axis=1)
# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

plt.scatter(train.GrLivArea, train.SalePrice, c = "blue", marker = "s")

plt.title("Looking for outliers")

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")
# Removing outliers

train = train[train.GrLivArea < 4000]
train['SalePrice'].describe()
plt.boxplot(train['SalePrice'], vert=False)
sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
#skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
train['SalePrice'] = np.log1p (train['SalePrice'])
sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
#scatter plot LotArea/saleprice

var = 'LotArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice');
#scatter plot TotRmsAbvGrd/saleprice

var = 'TotRmsAbvGrd'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice');
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#box plot MSSubClass/saleprice

var = 'MSSubClass'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#box plot MSZoning/saleprice

var = 'MSZoning'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#box plot Neighborhood/saleprice

var = 'Neighborhood'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 12))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#box plot BldgType/saleprice

var = 'BldgType'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#box plot OverallCond/saleprice

var = 'OverallCond'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#box plot SaleCondition/saleprice

var = 'SaleCondition'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
ntrain = train.shape[0]

ntest = test.shape[0]

y_trainf = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
#missing data

total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(35)
all_data['PoolQC'].value_counts()
all_data['PoolArea'].value_counts()
all_data[['PoolArea','PoolQC']].loc[all_data['PoolArea'] > 0]
all_data.loc[all_data.index[2416],'PoolQC'] = 'Gd'

all_data.loc[all_data.index[2499],'PoolQC'] = 'Gd'

all_data.loc[all_data.index[2595],'PoolQC'] = 'Gd'
all_data[['PoolArea','PoolQC']].loc[all_data['PoolArea'] > 0]
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data['PoolQC'].value_counts()
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
#box plot overallqual/saleprice

var = 'Neighborhood'

data = pd.concat([all_data['LotFrontage'], all_data[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 12))

fig = sns.boxplot(x=var, y="LotFrontage", data=data)

#fig.axis(ymin=0, ymax=800000);
all_data["GarageType"] = all_data["GarageType"].fillna("None")

all_data["GarageFinish"] = all_data["GarageFinish"].fillna("None")

all_data["GarageQual"] = all_data["GarageQual"].fillna("None")

all_data["GarageCond"] = all_data["GarageCond"].fillna("None")
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)
all_data[['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']].loc[pd.isnull(all_data['BsmtExposure']) | pd.isnull(all_data['BsmtCond']) | pd.isnull(all_data['BsmtQual']) | pd.isnull(all_data['BsmtFinType2']) | pd.isnull(all_data['BsmtFinType1']) ]
all_data.loc[pd.isnull(all_data['BsmtExposure']) & pd.isnull(all_data['BsmtCond']) & pd.isnull(all_data['BsmtQual']) & pd.isnull(all_data['BsmtFinType2']) & pd.isnull(all_data['BsmtFinType1']),['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']] = 'None'
all_data[['BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']].loc[pd.isnull(all_data['BsmtExposure']) | pd.isnull(all_data['BsmtCond']) | pd.isnull(all_data['BsmtQual']) | pd.isnull(all_data['BsmtFinType2']) | pd.isnull(all_data['BsmtFinType1']) ]
all_data['BsmtExposure'].value_counts()
all_data.loc[all_data.index[946],'BsmtExposure'] = 'No'

all_data.loc[all_data.index[1483],'BsmtExposure'] = 'No'

all_data.loc[all_data.index[2344],'BsmtExposure'] = 'No'
all_data['BsmtCond'].value_counts()
all_data.loc[all_data.index[2036],'BsmtCond'] = 'TA'

all_data.loc[all_data.index[2181],'BsmtCond'] = 'TA'

all_data.loc[all_data.index[2520],'BsmtCond'] = 'TA'
all_data['BsmtQual'].value_counts()
all_data.loc[all_data.index[2213],'BsmtQual'] = 'Gd'

all_data.loc[all_data.index[2214],'BsmtQual'] = 'Gd'
all_data['BsmtFinType2'].value_counts()
all_data.loc[all_data.index[332],'BsmtFinType2'] = 'Unf'
all_data[['MasVnrArea','MasVnrType']].loc[pd.isnull(all_data['MasVnrArea']) | pd.isnull(all_data['MasVnrType'])]
all_data['MasVnrArea'].value_counts()
all_data['MasVnrType'].value_counts()
all_data.loc[(pd.isnull(all_data['MasVnrArea']) & pd.isnull(all_data['MasVnrType'])),['MasVnrType']] = 'None'

all_data.loc[(pd.isnull(all_data['MasVnrArea'])),['MasVnrArea']] = 0

all_data.loc[(all_data['MasVnrArea'] != 0) & (all_data['MasVnrType'] == 'None'),['MasVnrType']] = 'BrkFace'

all_data.loc[all_data.index[2606],'MasVnrType'] = 'BrkFace'
all_data['MasVnrArea'].value_counts()
all_data['MasVnrType'].value_counts()
all_data[['MasVnrArea','MasVnrType']].loc[pd.isnull(all_data['MasVnrArea']) | pd.isnull(all_data['MasVnrType'])]
all_data[['MSZoning','Neighborhood']].loc[pd.isnull(all_data['MSZoning'])]
all_data['MSZoning'].loc[all_data['Neighborhood'] == 'IDOTRR'].value_counts()
all_data['MSZoning'].loc[all_data['Neighborhood'] == 'Mitchel'].value_counts()
all_data.loc[all_data.index[1911],'MSZoning'] = 'RM'

all_data.loc[all_data.index[2212],'MSZoning'] = 'RM'

all_data.loc[all_data.index[2246],'MSZoning'] = 'RM'

all_data.loc[all_data.index[2900],'MSZoning'] = 'RL'
all_data['MSZoning'].loc[all_data['Neighborhood'] == 'IDOTRR'].value_counts()
all_data['MSZoning'].loc[all_data['Neighborhood'] == 'Mitchel'].value_counts()
all_data[['BsmtHalfBath','BsmtFullBath','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2', 'BsmtExposure', 'BsmtCond', 'BsmtQual']].loc[pd.isnull(all_data['BsmtHalfBath']) | pd.isnull(all_data['BsmtFullBath']) | pd.isnull(all_data['TotalBsmtSF']) | pd.isnull(all_data['BsmtUnfSF']) | pd.isnull(all_data['BsmtFinSF1']) | pd.isnull(all_data['BsmtFinSF2'])]
all_data.loc[all_data.index[2116],['BsmtHalfBath','BsmtFullBath','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2']] = 0

all_data.loc[all_data.index[2184],['BsmtHalfBath','BsmtFullBath','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2']] = 0
all_data['Utilities'].value_counts()
all_data['Utilities'].loc[pd.isnull(all_data['Utilities'])] = 'AllPub'
all_data['Utilities'].value_counts()
all_data['Functional'].value_counts()
all_data['Functional'].loc[pd.isnull(all_data['Functional'])] = 'Typ'
all_data['Functional'].value_counts()
all_data['Electrical'].value_counts()
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['Electrical'].value_counts()
all_data['Exterior1st'].value_counts()
all_data['Exterior2nd'].value_counts()
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['KitchenQual'].value_counts()
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data[['GarageArea','GarageCars','GarageQual','GarageCond']].loc[pd.isnull(all_data['GarageArea']) | pd.isnull(all_data['GarageCars'])]
for col in ('GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
all_data['SaleType'].value_counts()
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



all_data['MoSold'] = all_data['MoSold'].astype(str)
#all_data = all_data.drop(['Utilities', 'Street', 'PoolQC',], axis=1)



all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']

all_data['TotalSF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] +

                                 all_data['1stFlrSF'] + all_data['2ndFlrSF'])



all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +

                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))



all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +

                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +

                              all_data['WoodDeckSF'])
all_data['haspool'] = all_data['PoolArea'].apply(lambda x: '1' if x > 0 else '0')

all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: '1' if x > 0 else '0')

all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: '1' if x > 0 else '0')

all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: '1' if x > 0 else '0')

all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: '1' if x > 0 else '0')
numeric_feats = all_data.dtypes[all_data.dtypes != object].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skew=all_data.select_dtypes(include=['int','float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skew_df=pd.DataFrame({'Skew':skew})

skewed_df=skew_df[(skew_df['Skew']>0.5)|(skew_df['Skew']<-0.5)]

skewed_df.index
lam=0.1

for col in ('MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch',

       'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch',

       'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF',

       'LotFrontage', 'GrLivArea', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces',

       'HalfBath', 'TotalBsmtSF', 'BsmtFullBath', 'OverallCond', 'YearBuilt',

       'GarageYrBlt'):

    all_data[col]=boxcox1p(all_data[col],boxcox_normmax(all_data[col] + 1))
skewness
all_data.describe()
all_data
# Encode some categorical features as ordered numbers when there is information in the order

all_data = all_data.replace({"BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "BsmtExposure" : {"None" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                        "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

                       "Street" : {"Grvl" : 1, "Pave" : 2},

                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}

                     )
final_features = pd.get_dummies(all_data).reset_index(drop=True)

final_features.shape
y = final_features['LotFrontage'].dropna()

X = final_features.dropna().drop('LotFrontage', axis=1)

X_predict = final_features.loc[pd.isnull(final_features['LotFrontage'])].drop('LotFrontage', axis=1)
X.shape
X_predict.shape
y.shape
def rmse(targets, predictions):

    return np.sqrt(mean_squared_error(targets,predictions))

rmse_score = make_scorer (rmse, greater_is_better = False)
scaler = RobustScaler ()

X = scaler.fit_transform(X)

X_predict = scaler.transform(X_predict)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 45)
params = {'alpha' : [0.01, 0.1, 1, 10, 100, 1000] }

gridsearch_lasso = GridSearchCV (Lasso(),params, scoring = rmse_score)

gridsearch_lasso.fit(X_train, y_train)

print ("Meilleurs parametres: ", gridsearch_lasso.best_params_)

gridsearch_lasso.score(X_test, y_test)
params = {'alpha' : [0.01, 0.1, 1, 10, 100, 1000] }

gridsearch_ridge = GridSearchCV (Ridge(),params, scoring = rmse_score)

gridsearch_ridge.fit(X_train, y_train)

print ("Meilleurs parametres: ", gridsearch_ridge.best_params_)

gridsearch_ridge.score(X_test, y_test)
params = {'alpha' : [0.01, 0.1, 1, 10, 100, 1000], 

          'l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]}

gridsearch_elasticnet = GridSearchCV (ElasticNet(),params, scoring = rmse_score)

gridsearch_elasticnet.fit(X_train, y_train)

print ("Meilleurs parametres: ", gridsearch_elasticnet.best_params_)

gridsearch_elasticnet.score(X_test, y_test)
rdforest = RandomForestRegressor(bootstrap = True, max_depth = 100, max_features = 3, min_samples_leaf = 3, min_samples_split = 8, n_estimators = 200)

rdforest.fit(X_train,y_train)

print ("R2 score for train set : ", rdforest.score(X_train, y_train))

print ("R2 score for test set : ", rdforest.score(X_test, y_test))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

KRR.fit(X_train, y_train)

y_pred = KRR.predict(X_test)

KRR.score (X_test, y_test)
svr = SVR(kernel='rbf', C=100, gamma=0.001)

svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)

svr.score(X_test, y_test)
final_features.loc[final_features["LotFrontage"].isnull(),"LotFrontage"] = gridsearch_elasticnet.predict(X_predict)

#y_predict = gridsearch_elasticnet.predict(X_predict)

#deter_data.loc[df[feature].isnull(), "Det" + feature] = model.predict(df[parameters])[df[feature].isnull()]
#missing data

total = final_features.isnull().sum().sort_values(ascending=False)

percent = (final_features.isnull().sum()/final_features.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(35)
scaler_lotfrontage = RobustScaler()

final_features = scaler_lotfrontage.fit_transform(final_features)
train = final_features[:ntrain]

test = final_features[ntrain:]
def rmse(targets, predictions):

    return np.sqrt(mean_squared_error(targets,predictions))

rmse_score = make_scorer (rmse, greater_is_better = False)
train.shape
y_trainf.shape
X_tr, X_val, y_tr, y_val = train_test_split(train, y_trainf, random_state = 3)
params = {'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] }

gridsearch_lasso = GridSearchCV (Lasso(),params, scoring=rmse_score)

gridsearch_lasso.fit(X_tr, y_tr)

print ("Meilleurs parametres: ", gridsearch_lasso.best_params_)

gridsearch_lasso.score(X_val, y_val)
params = {'alpha' : [0.0008, 0.009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006] }

gridsearch_lasso = GridSearchCV (Lasso(),params, scoring=rmse_score)

gridsearch_lasso.fit(X_tr, y_tr)

print ("Meilleurs parametres: ", gridsearch_lasso.best_params_)

gridsearch_lasso.score(X_val, y_val)
lasso = Lasso(alpha = 0.001)

lasso.fit(train, y_trainf)

lasso_pred = np.expm1(lasso.predict(test))

sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = lasso_pred

sub.to_csv('submission.csv',index=False)
sub