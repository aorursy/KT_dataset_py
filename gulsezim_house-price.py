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
import pandas as pd

import numpy as np



# Importing plotting libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

flatui = [ "#3498db", "#9b59b6", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

sns.set(style='white', context='notebook', palette=flatui)



# Importing statistics libraries

from scipy import stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats.stats import pearsonr, spearmanr



# Importing ML libraries

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb



# Muting warning messages

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print("Train Dataset: " + str(train.shape))

print("Test Dataset: " + str(test.shape))

train.head()
test_ID = test["Id"]



# Dropping the column "Id", since it's a non-explanatory variable.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)



print("Train Dataset: " + str(train.shape))

print("Test Dataset: " + str(test.shape))
print(train.columns.size)

train.columns
train['SalePrice'].describe()
plt.subplots(figsize=(15, 5))



plt.subplot(1, 2, 1)

g = sns.distplot(train['SalePrice'], fit=norm, label = "Skewness : %.2f"%(train['SalePrice'].skew()));

g = g.legend(loc="top-right")



plt.subplot(1, 2, 2)

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
train["SalePrice"] = np.log(train["SalePrice"])



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])



#Plotting the Distribution and the QQPlot

plt.subplots(figsize=(15, 5))



plt.subplot(1, 2, 1)

g = sns.distplot(train['SalePrice'], fit=norm)

g.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='top-right')



plt.subplot(1, 2, 2)

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
corr = train.corr()

plt.subplots(figsize=(30, 30))

cmap = sns.diverging_palette(10, 150, n=9, as_cmap=True, center="light")

sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1.0, center=0.2, square=True, linewidths=0, cbar_kws={"shrink": .5}, annot = True);
sns.lmplot('GrLivArea', 'SalePrice', train, size=5, aspect=2)


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<250000)].index)



sns.lmplot('GrLivArea', 'SalePrice', train, size=5, aspect=2)
plt.subplots(figsize=(15, 5))



plt.subplot(1, 2, 1)

g = sns.regplot(x=train['TotalBsmtSF'], y=train['SalePrice'], fit_reg=True).set_title("TotalBsmtFT")



plt.subplot(1, 2, 2)

g = sns.regplot(x=train['YearBuilt'], y=train['SalePrice'], fit_reg=True).set_title("YearBuilt")
train_length = len(train)

test_length = len(test)



# Saving the 'SalePrice' column that is only included in the Train Dataset. We will remove it and append it again later.

y_train = train.SalePrice.values



# Concatenating the datasets

joint = pd.concat((train, test)).reset_index(drop=True)



# Dropping the 'SalePrice' column, because it has values only for the train dataset

joint.drop(['SalePrice'], axis=1, inplace=True)



print(joint.shape)

joint.head()
NAs = joint.isnull().sum()



# Filtering out the columns that don't have missing values

NAs = NAs.drop(NAs[NAs == 0].index).sort_values(ascending=False)



# Plotting the count bars to get an idea of the missing values each column has

NAs.plot(kind='bar', figsize =(17, 5))
col = ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", 

       "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", 

       "MasVnrType", "MSSubClass")



for i in col:

    joint[i] = joint[i].fillna("None")

col = ("GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", 

       "TotalBsmtSF", "MasVnrArea", "BsmtFullBath", "BsmtHalfBath")



for i in col:

    joint[i] = joint[i].fillna(0)
col = ("MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType", "Functional")



for i in col:

    joint[i] = joint[i].fillna(joint[i].mode()[0])
joint["GarageYrBlt"] = joint["GarageYrBlt"].fillna(joint["YearBuilt"])



# Fixing missing values for LotFrontage

joint["LotFrontage"] = joint.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



# Dropping the Utilities variable

joint = joint.drop(['Utilities'], axis=1)
print("# of missing values = " + str(joint.isnull().sum().sum()))

joint['TotalSF'] = joint['TotalBsmtSF'] + joint['1stFlrSF'] + joint['2ndFlrSF']



# Freshness: How old was the house when it was sold

joint['Freshness'] = joint['YrSold'] - joint['YearBuilt']
joint.dtypes
col = ("YrSold", "MoSold", "OverallCond")

for i in col:

    joint[i] = joint[i].astype(str)

    

joint['MSSubClass'] = joint['MSSubClass'].apply(str)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for i in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(joint[i].values)) 

    joint[i] = lbl.transform(list(joint[i].values))



# shape        

print('Shape joint: {}'.format(joint.shape))
joint["OverallQual^2"] = joint["OverallQual"] ** 2

joint["GrLivArea^2"] = joint["GrLivArea"] ** 2

joint["GarageCars^2"] = joint["GarageCars"] ** 2

joint["GarageArea^2"] = joint["GarageArea"] ** 2

joint["TotalBsmtSF^2"] = joint["TotalBsmtSF"] ** 2

joint["1stFlrSF^2"] = joint["1stFlrSF"] ** 2

joint["FullBath^2"] = joint["FullBath"] ** 2

joint["TotRmsAbvGrd^2"] = joint["TotRmsAbvGrd"] ** 2



# Cubic Transformation for the top numeric variables

joint["OverallQual^3"] = joint["OverallQual"] ** 3

joint["GrLivArea^3"] = joint["GrLivArea"] ** 3

joint["GarageCars^3"] = joint["GarageCars"] ** 3

joint["GarageArea^3"] = joint["GarageArea"] ** 3

joint["TotalBsmtSF^3"] = joint["TotalBsmtSF"] ** 3

joint["1stFlrSF^3"] = joint["1stFlrSF"] ** 3

joint["FullBath^3"] = joint["FullBath"] ** 3

joint["TotRmsAbvGrd^3"] = joint["TotRmsAbvGrd"] ** 3





# Square Root Transformation for the top numeric variables

joint["OverallQual-Sq"] = np.sqrt(joint["OverallQual"])

joint["GrLivArea-Sq"] = np.sqrt(joint["GrLivArea"])

joint["GarageCars-Sq"] = np.sqrt(joint["GarageCars"])

joint["GarageArea-Sq"] = np.sqrt(joint["GarageArea"])

joint["TotalBsmtSF-Sq"] = np.sqrt(joint["TotalBsmtSF"])

joint["1stFlrSF-Sq"] = np.sqrt(joint["1stFlrSF"])

joint["FullBath-Sq"] = np.sqrt(joint["FullBath"])

joint["TotRmsAbvGrd-Sq"] = np.sqrt(joint["TotRmsAbvGrd"])
numcolumns = joint.dtypes[joint.dtypes != "object"].index



# Check how skewed they are

skewed = joint[numcolumns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)



plt.subplots(figsize =(17, 8))

skewed.plot(kind='bar');
skewness = skewed[abs(skewed) > 0.75]



skewed = skewness.index

lam = 0.15

for i in skewed:

    joint[i] = boxcox1p(joint[i], lam)



print(skewness.shape[0],  "skewed numerical features have been Box-Cox transformed")


joint = pd.get_dummies(joint)

print(joint.shape)

joint.head()
train = joint[:train_length]

test = joint[train_length:]
print("Train Dataset: " + str(train.shape))

print("Test Dataset: " + str(test.shape))
n_folds = 10



def rmsle_cv(model):

    kfolds = KFold(n_folds, shuffle=True, random_state=13).get_n_splits(train.values)

    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kfolds))

    return(rmse)
lasso = Lasso(alpha =0.0005, random_state=1)

lasso.fit(train.values, y_train)

lasso_pred = np.exp(lasso.predict(test.values))

score = rmsle_cv(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
lasso1 = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

lasso1.fit(train.values, y_train)

lasso1_pred = np.exp(lasso1.predict(test.values))

score = rmsle_cv(lasso1)

print("Lasso (Robust Scaled) score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
elnet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=42)

elnet.fit(train.values, y_train)

elnet_pred = np.exp(elnet.predict(test.values))

score = rmsle_cv(elnet)

print("Elastic Net score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.model_selection import cross_val_score, KFold

from scipy.stats import sem

from sklearn.tree import DecisionTreeRegressor



all_depths = []

all_mean_scores = []

for max_depth in range(1, 11):

    all_depths.append(max_depth)

    simple_tree = DecisionTreeRegressor(max_depth=max_depth)

    cv = KFold(n_splits=5, shuffle=True, random_state=13)

    scores = cross_val_score(simple_tree, train.values, y_train, cv=cv)

    mean_score = np.mean(scores)

    all_mean_scores.append(np.mean(scores))

    print("max_depth = ", max_depth, scores, mean_score, sem(scores))
plt.plot(all_depths, all_mean_scores, label='True y')

plt.xlabel('max depth')

plt.ylabel('mean score')
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=5)

tree.fit(train.values, y_train)
dtree_pred = np.exp(tree.predict(test.values))
score = rmsle_cv(tree)

print("Decision Tree score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.ensemble import RandomForestRegressor



# Building the Random Forest Regressor

rfr = RandomForestRegressor(max_depth=None, random_state=0, min_samples_split=2, 

                              n_estimators=100)

rfr.fit(train.values, y_train)

rfr_pred = np.exp(rfr.predict(test.values))

score = rmsle_cv(rfr)

print("Random Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
xg_reg = xgb.XGBRegressor(objective ='reg:linear', 

                          colsample_bytree = 0.5, 

                          learning_rate = 0.15,

                          max_depth = 4, 

                          reg_lambda = 0.5, 

                          n_estimators = 100)

xg_reg.fit(train.values, y_train)

xg_reg_pred = np.exp(xg_reg.predict(test.values))

score = rmsle_cv(xg_reg)

print("XGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05)

model_lgb.fit(train.values, y_train)

model_lgb_pred = np.exp(model_lgb.predict(test.values))

score = rmsle_cv(model_lgb)

print("LightGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn import model_selection

from sklearn.model_selection import train_test_split
x_trn, x_val, y_trn, y_val = model_selection.train_test_split(train.values, y_train, test_size=0.3, random_state=42)

print('x_trn: ', x_trn.shape, '\nx_val: ', x_val.shape, '\ny_trn: ', y_trn.shape, '\ny_val: ', y_val.shape)
stacked_validation = pd.DataFrame()

stacked_test = pd.DataFrame()



# Fitting XGBoost, LightGBM, ElasticNet and AdaBoost using the new test + validation dataset

xg_reg.fit(x_trn, y_trn)

model_lgb.fit(x_trn, y_trn)

elnet.fit(x_trn, y_trn)



# Making predictions for the validation dataset

predictions1 = xg_reg.predict(x_val)

predictions2 = model_lgb.predict(x_val)

predictions3 = elnet.predict(x_val)





# Making predictions for the actual test dataset

tpred1 = xg_reg.predict(test.values)

tpred2 = model_lgb.predict(test.values)

tpred3 = elnet.predict(test.values)





# Stacking the prediction outcomes in the 2 data frames

stacked_validation = np.column_stack((predictions1, predictions2, predictions3, predictions4))

stacked_test = np.column_stack((tpred1, tpred2, tpred3, tpred4))



# Setting LASSO1 as the meta-model and fitting it with the stacked predictions

lasso1.fit(stacked_validation, y_val)



# Making predictions for the actual test dataset, using the meta-model

final_predictions = np.exp(lasso1.predict(stacked_test))
stacked_validation = pd.DataFrame()

stacked_test = pd.DataFrame()



# Fitting XGBoost, LightGBM, ElasticNet and AdaBoost using the new test + validation dataset

xg_reg.fit(x_trn, y_trn)

lasso.fit(x_trn, y_trn)

elnet.fit(x_trn, y_trn)





# Making predictions for the validation dataset

predictions1 = xg_reg.predict(x_val)

predictions2 = model_lgb.predict(x_val)

predictions3 = elnet.predict(x_val)







# Making predictions for the actual test dataset

tpred1 = xg_reg.predict(test.values)

tpred2 = model_lgb.predict(test.values)

tpred3 = elnet.predict(test.values)





# Stacking the prediction outcomes in the 2 data frames

stacked_validation = np.column_stack((predictions1, predictions2, predictions3))

stacked_test = np.column_stack((tpred1, tpred2, tpred3))



# Setting LASSO1 as the meta-model and fitting it with the stacked predictions

model_lgb.fit(stacked_validation, y_val)



# Making predictions for the actual test dataset, using the meta-model

final_predictions = np.exp(model_lgb.predict(stacked_test))
ensemble = final_predictions*0.75 + xg_reg_pred*0.15 + model_lgb_pred*0.15
submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = final_predictions

submission.to_csv('House_price.csv',index=False)