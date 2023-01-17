import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams

import warnings



# ignore certain warnings

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=RuntimeWarning)



# set seaborn defaults

sns.set()



%config InlineBackend.figure_format = 'png' #set 'png' here when working in notebook

%matplotlib inline



# identify data sets

trainData = '../input/train.csv'

testData = '../input/test.csv'



# import data sets

train = pd.read_csv(trainData, header=0)

test = pd.read_csv(testData, header=0)



# combine all data (ignoring Id and SalePrice features)

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']))
# view training data

train.head()
# view testing data

test.head()
# view combined data

all_data.head()
rcParams['figure.figsize'] = (6.0, 6.0) # define size of figure

sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index).reset_index(drop=True)



# reset combined data set with new training set

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']))
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder



cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))
from scipy.stats import skew



# plot histogram of "SalePrice"

rcParams['figure.figsize'] = (12.0, 6.0) # define size of figure

g = sns.distplot(train["SalePrice"], label="Skewness: %.2f"%(train["SalePrice"].skew()))

g = g.legend(loc="best")

plt.show()
normalizedSalePrice = np.log1p(train["SalePrice"])



# plot histogram of log transformed "SalePrice"

rcParams['figure.figsize'] = (12.0, 6.0) # define size of figure

g = sns.distplot(normalizedSalePrice, label="Skewness: %.2f"%(normalizedSalePrice.skew()))

g = g.legend(loc="best")

plt.show()
# apply log transform to target

train["SalePrice"] = np.log1p(train["SalePrice"])
# determine features that are heavily skewed

def get_skewed_features():

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) # computes "skewness"

    skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]

    return skewed_feats.index
from sklearn.preprocessing import power_transform



# find heavily skewed numerical features

skewed_feats = get_skewed_features()

print("{} heavily skewed features.".format(len(skewed_feats)))



# apply power transform to all heavily skewed numeric features

all_data[skewed_feats] = power_transform(all_data[skewed_feats], method='yeo-johnson')

print("Applied power transform.")
# create dummy variables

all_data = pd.get_dummies(all_data)

all_data.shape # we now have 219 features columns compared to original 79
# check for any missing values

all_data.isnull().any().any()
# replace NA's with the mean of the feature

all_data = all_data.fillna(all_data.mean())



# check again for any missing values

all_data.isnull().any().any()
# create matrices for sklearn

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
from sklearn.model_selection import cross_val_score



# determine average root mean square error (RMSE) using k-fold cross validation

def rmse_cv(model, cv=5):

    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = cv))

    return rmse
from sklearn.linear_model import LinearRegression



# estimate RMSE for linear regression model

linearModel = LinearRegression()

rmse = rmse_cv(linearModel)

print("RMSE estimate: {}, std: {}".format(rmse.mean(), rmse.std()))
# fit linear model

linearModel.fit(X_train, y)



# get largest magnitude coefficients

coef = pd.Series(linearModel.coef_, index = X_train.columns)

imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])



rcParams['figure.figsize'] = (8.0, 10.0) # define size of figure

imp_coef.plot(kind = "barh")

plt.title("Most Important Coefficients Selected by Ridge")

plt.show()
from sklearn.linear_model import Ridge



# determine RMSE for ridge regression model with alpha = 0.1

ridgeModel = Ridge(alpha = 0.1)

rmse = rmse_cv(ridgeModel)

print("RMSE estimate: {}, std: {}".format(rmse.mean(), rmse.std()))
rcParams['figure.figsize'] = (12.0, 6.0) # define size of figure



# calculate RMSE over several alphas

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)



# plot RMSE vs alpha

cv_ridge.plot(title = "RMSE of Ridge Regression as Alpha Scales")

plt.xlabel("alpha")

plt.ylabel("rmse")

plt.show()
rcParams['figure.figsize'] = (12.0, 6.0) # define size of figure



# calculate RMSE over several alphas

alphas = np.linspace(9.8, 15.2, 541)

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)



# plot RMSE vs alpha

cv_ridge.plot(title = "RMSE of Ridge Regression as Alpha Scales")

plt.xlabel("alpha")

plt.ylabel("rmse")

plt.show()
optimalRidgeAlpha = cv_ridge[cv_ridge == cv_ridge.min()].index.values[0]

print("Optimal ridge alpha: {}".format(optimalRidgeAlpha))
# determine RMSE for ridge regression model with optimal alpha

ridgeModel = Ridge(alpha = optimalRidgeAlpha)

rmse = rmse_cv(ridgeModel)

print("RMSE estimate: {}, std: {}".format(rmse.mean(), rmse.std()))
# fit ridge model

ridgeModel.fit(X_train, y)



# get largest magnitude coefficients

ridge_coef = pd.Series(ridgeModel.coef_, index = X_train.columns)

ridge_imp_coef = pd.concat([ridge_coef.sort_values().head(10), ridge_coef.sort_values().tail(10)])



rcParams['figure.figsize'] = (8.0, 10.0) # define size of figure

df = pd.DataFrame({ "RidgeRegression" : ridge_imp_coef, "LinearRegression" : imp_coef })

df.plot(kind = "barh")

plt.title("Most Important Coefficients Selected by Ridge")

plt.show()
from sklearn.linear_model import Lasso



# determine RMSE for lasso regression model with alpha = 0.1

lassoModel = Lasso(alpha = 0.1)

rmse = rmse_cv(lassoModel)

print("RMSE estimate: {}, std: {}".format(rmse.mean(), rmse.std()))
from sklearn.linear_model import Lasso



rcParams['figure.figsize'] = (12.0, 6.0) # define size of figure



# calculate RMSE over several alphas

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]

cv_lasso = pd.Series(cv_lasso, index = alphas)



# plot RMSE vs alpha

cv_lasso.plot(title = "RMSE of Lasso Regression as Alpha Scales")

plt.xlabel("alpha")

plt.ylabel("rmse")

plt.show()
from sklearn.linear_model import LassoCV



# use built in LassoCV function to select best model for data

lassoModel = LassoCV(alphas = np.linspace(0.0002, 0.0022, 21), cv = 5).fit(X_train, y)

lassoModel.alpha_



optimalLassoAlpha = lassoModel.alpha_

print("Optimal lasso alpha: {}".format(optimalLassoAlpha))
lassoModel = Lasso(alpha = optimalLassoAlpha)

rmse = rmse_cv(lassoModel)

print("RMSE estimate: {}, std: {}".format(rmse.mean(), rmse.std()))
# fit lasso model

lassoModel.fit(X_train, y)



# get largest magnitude coefficients

lasso_coef = pd.Series(lassoModel.coef_, index = X_train.columns)

lasso_imp_coef = pd.concat([lasso_coef.sort_values().head(10), lasso_coef.sort_values().tail(10)])



rcParams['figure.figsize'] = (8.0, 10.0) # define size of figure

df = pd.DataFrame({ "LassoRegression" : lasso_imp_coef, "LinearRegression" : imp_coef })

df.plot(kind = "barh")

plt.title("Most Important Coefficients Selected by Lasso")

plt.show()
lasso_coef = pd.Series(lassoModel.coef_, index = X_train.columns)

print(sum(lasso_coef != 0))

print(sum(lasso_coef == 0))
# scale alpha

alphas = np.linspace(0.0002, 0.4002, 2001)

nonZeros = []



# for each alpha, fit model to training data

for alpha in alphas:

    lassoModel = Lasso(alpha = alpha).fit(X_train, y)

    coef = pd.Series(lassoModel.coef_, index = X_train.columns)

    # append the number of non-zero coefficients

    nonZeros = np.append(nonZeros, sum(coef != 0))



# plot number of non-zeros (L0-Norm) vs alpha

rcParams['figure.figsize'] = (12.0, 6.0) # define size of figure

lzeroNorm = pd.Series(nonZeros, index = alphas)

lzeroNorm.plot(title = "L0-Norm of Lasso Regression Model as Alpha Scales")

plt.xlabel("alpha")

plt.ylabel("number of non-zeros")

plt.show()
lzeroNorm.max()
lzeroNorm.min()
linearModel = LinearRegression().fit(X_train, y)

lr_submission = pd.DataFrame()

lr_submission['Id'] = test['Id']

lr_submission['SalePrice'] = np.expm1(linearModel.predict(X_test))

lr_submission.to_csv('linear-regression.csv', index=False)
ridgeModel = Ridge(alpha = optimalRidgeAlpha).fit(X_train, y)

ridge_submission = pd.DataFrame()

ridge_submission['Id'] = test['Id']

ridge_submission['SalePrice'] = np.expm1(ridgeModel.predict(X_test))

ridge_submission.to_csv('ridge.csv', index=False)
lassoModel = Lasso(alpha = optimalLassoAlpha).fit(X_train, y)

lasso_submission = pd.DataFrame()

lasso_submission['Id'] = test['Id']

lasso_submission['SalePrice'] = np.expm1(lassoModel.predict(X_test))

lasso_submission.to_csv('lasso.csv', index=False)