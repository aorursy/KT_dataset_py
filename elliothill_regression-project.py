# Import necessary packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt



%config InlineBackend.figure_format = 'png'

%matplotlib inline
# Read in the training and testing data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Look at the training data

train.head()
# Summarize the training data

train.describe()
# Look at the testing data

test.head()
# Summarize the testing data

test.describe()
# Look at the number of rows and columns for each data set

train.shape,test.shape
# Look for duplicate entries

idsUnique = len(set(train.Id))

idsTotal = train.shape[0]

idsdupe = idsTotal - idsUnique

print(idsdupe)
# Look for null values by feature

train_null = train.isnull().sum()

train_null = train_null[train_null>0]

train_null = train_null.sort_values(ascending=False)

train_null = pd.DataFrame({"NullValues": train_null})



# Plot null values by feature

sns.barplot(x=train_null['NullValues'],y=train_null.index)

plt.xlabel('Null Values', fontsize=15)

plt.ylabel('Feature', fontsize=15)

plt.title('Null Values by Feature', fontsize=15)

# Correlation between all variables



# Compute the correlation matrix

corr = train.corr()



# Generate a mask for the upper triangle (We only want to see the lower triangle in this plot)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

sns.set(style="white")

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.xlabel('Feature', fontsize=15)

plt.ylabel('Feature', fontsize=15)

plt.title('Correlation between Features', fontsize=15)

# Look at the skewness of the SalePrice

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)

sns.distplot(train['SalePrice'], ax=ax1)

sns.distplot(np.log1p(train['SalePrice']), ax=ax2)

plt.xlabel('log(SalePrice+1)')



# Take the log of the SalePrice because it is skewed left

train["SalePrice"] = np.log1p(train["SalePrice"])
# Combine training and testing data

# Get rid of Id and SalePrice columns

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))

# Fill in null values



# Null values for these features mean that there is none

# e.g. NA for PoolQC means there is no pool

all_data['PoolQC'] = all_data['PoolQC'].fillna('None')

all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')

all_data['Alley'] = all_data['Alley'].fillna('None')

all_data['BsmtQual'] = all_data['BsmtQual'].fillna('None')

all_data['BsmtCond'] = all_data['BsmtCond'].fillna('None')

all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna('None')

all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna('None')

all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna('None')

all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')

all_data['GarageType'] = all_data['GarageType'].fillna('None')

all_data['GarageFinish'] = all_data['GarageFinish'].fillna('None')

all_data['GarageQual'] = all_data['GarageQual'].fillna('None')

all_data['GarageCond'] = all_data['GarageCond'].fillna('None')

all_data['Fence'] = all_data['GarageCond'].fillna('None')



# Null values for these features should get filled with 0

# e.g. NA for PoolArea means there is no pool and therefore the pool area is 0 SF

for col in ('MiscVal','PoolArea','ScreenPorch','3SsnPorch','EnclosedPorch','OpenPorchSF','WoodDeckSF','BsmtFullBath','BsmtHalfBath','HalfBath','LowQualFinSF','1stFlrSF','2ndFlrSF','GarageYrBlt', 'GarageArea', 'GarageCars','TotalBsmtSF','Fireplaces','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'):

    all_data[col] = all_data[col].fillna(0)



# Null values for these features are filled with the most frequently seen value for that feature

# e.g. every house probably has at least one FullBath so we fill it with the most common value

for col in ('FullBath','BedroomAbvGr','KitchenAbvGr','SaleType','SaleCondition'):

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])



# Null values for these features probably follow median values for that neighborhood

# e.g. the house in a neighboorhood can be assumed to be built at the median time 

# of all the other houses in the neighboorhood

for col in ('LotFrontage','LotArea','YearBuilt','GarageYrBlt'):    

    all_data[col] = all_data.groupby("Neighborhood")[col].transform(

        lambda x: x.fillna(x.median()))

    

# Null values for these features are assumed to follow the most frequently obvserved values within that neighboorhood

# e.g. A house is assumed to have a PavedDrive value similar to most other houses in the neighborhood

for col in ('PavedDrive','MSSubClass','CentralAir','Electrical','GarageType','MSZoning','Street','Utilities','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd'):    

    all_data[col] = all_data.groupby("Neighborhood")[col].transform(

        lambda x: x.fillna(x.mode()[0]))
# Feature engineering

# We can combine related features

all_data['TotalBsmtFinSF'] = all_data['BsmtFinSF1']+all_data['BsmtFinSF2']
# Look at the skewness of the independent variables



# Select numerical variables

numerical_features = all_data.select_dtypes(exclude = ["object"]).columns

num_feats = all_data[numerical_features]



# Import necessary package

from scipy.stats import skew 



# Determine skewness of variables

skewness = num_feats.apply(lambda x: skew(x))

skewness = skewness[skewness>0.5]

skewness.sort_values(ascending=False)

skewed_feats = skewness.index



# Take the log of highly skewed variables

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# Get dummy variables for categorical features

all_data = pd.get_dummies(all_data)

# Fill all remaining null values with mean of the feature

all_data = all_data.fillna(all_data.mean())
# Separate all_data into training and testing sets for modeling

train_X = all_data[:train.shape[0]]

test_X = all_data[train.shape[0]:]

y = train['SalePrice']
# Create function to generate Kfold Cross-validation and determine RMSE

from sklearn.model_selection import KFold,cross_val_score



n_folds = 5



def rmse_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=50).get_n_splits(train_X)

    rmse= np.sqrt(-cross_val_score(model, train_X, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
# Get a basline RMSE for analysis

# Test LinearRegression base model

from sklearn.linear_model import LinearRegression

baseline = LinearRegression()

score = rmse_cv(baseline)

print(score.mean())
# Test Ridge base model

from sklearn.linear_model import Ridge

ridge = Ridge()

score = rmse_cv(ridge)

print(score.mean())
# Test RidgeCV base model

from sklearn.linear_model import RidgeCV



ridgecv = RidgeCV()

score = rmse_cv(ridgecv)

print(score.mean())
# Test KernelRidge base model

from sklearn.kernel_ridge import KernelRidge



KRR = KernelRidge()

score = rmse_cv(KRR)

print(score.mean())
# Test BayesianRidge base model

from sklearn.linear_model import BayesianRidge



BR = BayesianRidge()

score = rmse_cv(BR)

print(score.mean())
# Test Lasso base model

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.0005)

score = rmse_cv(lasso)

print(score.mean())
## Test LassoCV base model

from sklearn.linear_model import LassoCV

lassocv = LassoCV()

score = rmse_cv(lassocv)

print(score.mean())
# Test DecisionTreeRegressor base model

from sklearn.tree import DecisionTreeRegressor

DTR = DecisionTreeRegressor()

score = rmse_cv(DTR)

print(score.mean())
# Test GradientBoostingRegressor base model

from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor()

score = rmse_cv(GBR)

print(score.mean())
# Test XGBoost base model

import xgboost as xgb



XGB = xgb.XGBRegressor()

score = rmse_cv(XGB)

print(score.mean())
# Test LightGBM base model

import lightgbm as lgb



LGB = lgb.LGBMRegressor()

score = rmse_cv(LGB)

print(score.mean())
# Determine which loss function to use

l = ['ls','lad','huber','quantile']

cv_ridge = [rmse_cv(GradientBoostingRegressor(loss=loss)).mean()

            for loss in l]

cv_ridge = pd.Series(cv_ridge, index = l)

cv_ridge.plot(title = "Loss")

plt.xlabel("Loss Function")

plt.ylabel("rmse")
# Determine optimal learning rate

l = [0.001,0.01,0.1,0.2,0.3,0.4,0.5]

cv_ridge = [rmse_cv(GradientBoostingRegressor(loss='ls',learning_rate=learn)).mean()

            for learn in l]

cv_ridge = pd.Series(cv_ridge, index = l)

cv_ridge.plot(title = "Learning Rate")

plt.xlabel("learning_rate")

plt.ylabel("rmse")
# Determine optimal number of boosting stages

n_estimators = [50,100,200,300,400,500,600,700,800,900,1000]

cv_ridge = [rmse_cv(GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators = n)).mean() 

            for n in n_estimators]

cv_ridge = pd.Series(cv_ridge, index = n_estimators)

cv_ridge.plot(title = "Number of Boosting Stages")

plt.xlabel("n_estimators")

plt.ylabel("rmse")
# Determine optimal max depth

depths = [1,2,3,4,5]

cv_ridge = [rmse_cv(GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators = 400,max_depth = d)).mean() 

            for d in depths]

cv_ridge = pd.Series(cv_ridge, index = depths)

cv_ridge.plot(title = "Max Depths")

plt.xlabel("max_depth")

plt.ylabel("rmse")
# Determine optimal number of minimum samples needed to split a node

min_split = [2,3,4,5]

cv_ridge = [rmse_cv(GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators = 400,max_depth = 4,min_samples_split = m)).mean() 

            for m in min_split]

cv_ridge = pd.Series(cv_ridge, index = min_split)

cv_ridge.plot(title = "Minimum Number of Samples to Split a Node")

plt.xlabel("min_samples_split")

plt.ylabel("rmse")
# Determine optimal number of minimum samples to a leaf node

min_leaf = [1,2,3,4,5]

cv_ridge = [rmse_cv(GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators = 400,max_depth = 4,min_samples_split=2,min_samples_leaf = m)).mean() 

            for m in min_leaf]

cv_ridge = pd.Series(cv_ridge, index = min_leaf)

cv_ridge.plot(title = "Minimum Number of Samples to a Leaf Node")

plt.xlabel("min_samples_leaf")

plt.ylabel("rmse")
# Combine optimal model parameters into model

from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators = 400,max_depth = 4,min_samples_split=2)

score = rmse_cv(GBR)

print(score.mean())
# Determine optimal alpha

alphas = [1.0,0.1,0.05,0.005,0.0005]

cv_ridge = [rmse_cv(Lasso(alpha=alpha)).mean()

            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Alpha")

plt.xlabel("Alpha")

plt.ylabel("rmse")
# Determine whether or not to fit intercept

fit = [True,False]

cv_ridge = [rmse_cv(Lasso(alpha=0.0005,fit_intercept = f)).mean()

            for f in fit]

cv_ridge = pd.Series(cv_ridge, index = fit)

cv_ridge.plot(title = "Fit Intercept")

plt.xlabel("fit_intercept")

plt.ylabel("rmse")
# Determine whether or not to normalize

norm = [True,False]

cv_ridge = [rmse_cv(Lasso(alpha=0.0005,fit_intercept = True,normalize = n)).mean()

            for n in norm]

cv_ridge = pd.Series(cv_ridge, index = norm)

cv_ridge.plot(title = "Normalize")

plt.xlabel("normalize")

plt.ylabel("rmse")
# Combine optimal parameters into model

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.0005,fit_intercept = True,normalize = False)

score = rmse_cv(lasso)

print(score.mean())
# Generate submission for Gradient Boosting model

pred_GBR = GBR.fit(train_X,y).predict(test_X)

pred_GBR = np.expm1(pred_GBR)

submission_GBR = pd.DataFrame({"Id":test.Id, "SalePrice":pred_GBR})

submission_GBR.to_csv("GBM_sol.csv",index=False)



# Generate submission for Lasso model

pred_Lasso = GBR.fit(train_X,y).predict(test_X)

pred_Lasso = np.expm1(pred_Lasso)

submission_Lasso = pd.DataFrame({"Id":test.Id, "SalePrice":pred_Lasso})

submission_Lasso.to_csv("Lasso_sol.csv",index=False)



# Combine model predictions with different weights

pred_combo1 = 0.3*pred_GBR+0.7*pred_Lasso

submission_combo1 = pd.DataFrame({"Id":test.Id, "SalePrice":pred_combo1})

submission_combo1.to_csv("combo1_sol.csv",index=False)



pred_combo2 = 0.5*pred_GBR+0.5*pred_Lasso

submission_combo2 = pd.DataFrame({"Id":test.Id, "SalePrice":pred_combo2})

submission_combo2.to_csv("combo2_sol.csv",index=False)



pred_combo3 = 0.7*pred_GBR+0.3*pred_Lasso

submission_combo3 = pd.DataFrame({"Id":test.Id, "SalePrice":pred_combo3})

submission_combo3.to_csv("combo3_sol.csv",index=False)
