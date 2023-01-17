import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt

%matplotlib inline
import warnings

warnings.filterwarnings('ignore')
traindata = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

testdata = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
traindata.info()
#Before dropping the outliers

fig, ax = plt.subplots()

ax.scatter(x = traindata['GrLivArea'], y = traindata['SalePrice'])

plt.ylabel('SalePrice')

plt.xlabel('GrLivArea')

plt.title("Before dropping the outliers")

plt.show()



#Deleting outliers

traindata = traindata.drop(traindata[(traindata['GrLivArea']>4000) & (traindata['SalePrice']<300000)].index)



#After dropping the outliers

fig, ax = plt.subplots()

ax.scatter(traindata['GrLivArea'], traindata['SalePrice'])

plt.ylabel('SalePrice')

plt.xlabel('GrLivArea')

plt.title("After dropping the outliers")

plt.show()
traindata['SalePrice'].describe()
#Histogram

sns.distplot(traindata['SalePrice'])



#QQ-Plot

fig = plt.figure()

res = stats.probplot(traindata['SalePrice'], plot=plt)
print("Skewness before transformation:", round(traindata['SalePrice'].skew(),5))

print("Kurtosis before transformation:", round(traindata['SalePrice'].kurt(),5))
traindata['SalePrice'] = np.log(traindata['SalePrice'])
#Histogram

sns.distplot(traindata['SalePrice'])



#QQ-Plote

fig = plt.figure()

res = stats.probplot(traindata['SalePrice'], plot=plt)
print("Skewness after transformation:", round(traindata['SalePrice'].skew(),5))

print("Kurtosis after transformation:", round(traindata['SalePrice'].kurt(),5))
ntrain = len(traindata)

y = traindata['SalePrice']



combined_df = pd.concat([traindata,testdata], ignore_index=True)

combined_df.drop(columns= 'SalePrice', inplace = True)

print("Shape of combined data frame: ", combined_df.shape)
#This heatmap shows the missing data present in the data frame

plt.figure(figsize=(20,3))

sns.heatmap(combined_df.isnull(), yticklabels=False, cbar=False, cmap = 'viridis')
(combined_df.isnull().sum().sort_values(ascending = False) / len(combined_df)) * 100
combined_df['PoolQC'] = combined_df['PoolQC'].fillna('None')

combined_df['MiscFeature'] = combined_df['MiscFeature'].fillna('None')

combined_df['Alley'] = combined_df['Alley'].fillna('None')

combined_df['Fence'] = combined_df['Fence'].fillna('None')

combined_df['FireplaceQu'] = combined_df['FireplaceQu'].fillna('None')
combined_df['LotFrontage'] = combined_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for columns in ('GarageType','GarageFinish', 'GarageQual','GarageCond','MSSubClass','BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType'):

    combined_df[columns] = combined_df[columns].fillna('None')
for columns in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):

    combined_df[columns] = combined_df[columns].fillna(0)
for columns in ('MSZoning','Electrical', 'KitchenQual','Exterior1st','Exterior2nd','SaleType'):

    combined_df[columns] =  combined_df[columns].fillna(combined_df[columns]).mode()[0]
combined_df['Functional'] =  combined_df['Functional'].fillna('Typ')

combined_df.drop(columns='Utilities', axis = 1, inplace = True)
(combined_df.isnull().sum().sort_values(ascending = False) / len(combined_df)) * 100
combined_df['MSSubClass'] =  combined_df['MSSubClass'].apply(str)

combined_df['OverallCond'] = combined_df['OverallCond'].astype(str)
combined_df['TotalSF'] = combined_df['TotalBsmtSF'] + combined_df['1stFlrSF'] + combined_df['2ndFlrSF']
#Getting only the numerical features

numeric_features = combined_df.dtypes[combined_df.dtypes != 'object'].index



#Checking the skewness of all the numerical features

skewed_features = combined_df[numeric_features].apply(lambda x: x.skew()).sort_values(ascending = False)

skewed_df = pd.DataFrame({'Skew': skewed_features})

display(skewed_df)
from scipy.special import boxcox1p



skewed_df = skewed_df[abs(skewed_df) > 0.75]

skewed_feats = skewed_df.index

lam = 0.15

for feats in skewed_feats:

    combined_df[feats] = boxcox1p(combined_df[feats], lam)
combined_df = pd.get_dummies(combined_df, drop_first=True)

combined_df.shape
X_train = combined_df[:traindata.shape[0]]

X_test = combined_df[traindata.shape[0]:]

y = traindata.SalePrice
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse.mean())
from sklearn.linear_model import LinearRegression



lm_rmse = rmse_cv(LinearRegression())

print("RMSE for Linear Regression: ", lm_rmse)
from sklearn.linear_model import Lasso, LassoCV



lassocv = LassoCV(cv=5,random_state=1)

lassocv.fit(X_train, y)

best_alpha = lassocv.alpha_



lasso_model = make_pipeline(RobustScaler(), Lasso(alpha= best_alpha, random_state=1))

lasso_rmse = rmse_cv(lasso_model)

print("RMSE for LASSO (L1 Regularization): ", lasso_rmse)
from sklearn.linear_model import Ridge, RidgeCV



ridge_model = make_pipeline(RobustScaler(), RidgeCV(alphas=np.logspace(-10,10,100)))

ridge_rmse = rmse_cv(ridge_model)

print("RMSE for Ridge Regression (L2 Regularization): ", ridge_rmse)
from sklearn.linear_model import ElasticNet, ElasticNetCV



elasticnet_cv = ElasticNetCV(l1_ratio=np.arange(0.1,1,0.1), cv=5, random_state=1)

elasticnet_cv.fit(X_train, y)

best_l1_ratio = elasticnet_cv.l1_ratio_

best_aplha = elasticnet_cv.alpha_



elasticnet_model = make_pipeline(RobustScaler(), ElasticNet(alpha=best_alpha, l1_ratio= best_l1_ratio, random_state=1))

elasticnet_rmse = rmse_cv(elasticnet_model)

print("RMSE for Elastic Net(L1 and L2 Regularization): ", elasticnet_rmse)
from sklearn.ensemble import RandomForestRegressor



randomforest_model = rmse_cv(RandomForestRegressor(random_state=1))

print("RMSE for Random Forest: ", randomforest_model)
from sklearn.ensemble import GradientBoostingRegressor



gradientboost_model = GradientBoostingRegressor(learning_rate=0.1, loss='huber', n_estimators=3000, random_state=1)

gradientboost_cv = rmse_cv(gradientboost_model)

print("RMSE for Gradient Boost: ", gradientboost_cv)
from sklearn.ensemble import AdaBoostRegressor



adaboost_lassomodel = AdaBoostRegressor(lasso_model, n_estimators=50, learning_rate=0.001, random_state=1)

adaboost_lassocv = rmse_cv(adaboost_lassomodel)

print("RMSE for Adaptive Boosting with LASSO estimator: ", adaboost_lassocv)
from sklearn.ensemble import AdaBoostRegressor



adaboost_enetmodel = AdaBoostRegressor(elasticnet_model, n_estimators=50, learning_rate=0.001, random_state=1)

adaboost_enetcv = rmse_cv(adaboost_enetmodel)

print("RMSE for Adaptive Boosting with Elastic Net estimator: ", adaboost_enetcv)
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler



pca = make_pipeline(RobustScaler(), PCA(n_components = 3, random_state = 1))

pca.fit(X_train.transpose())

print(f"Proportion of variance explained by the components: {pca.steps[1][1].explained_variance_ratio_}")



# we are using 3 components in this case

p_comps = pca.steps[1][1].components_.transpose()



pca_lm_rmse = np.sqrt(-cross_val_score(LinearRegression(), p_comps, y, cv = 5, scoring = "neg_mean_squared_error")).mean()

print(f"RMSE for Linear Regression after PCA reduction: [{pca_lm_rmse}]")
adaboost_enetmodel.fit(X_train, np.exp(y))

submission_predictions = adaboost_enetmodel.predict(X_test)
results = pd.DataFrame({'Id': testdata['Id'], 'SalePrice':submission_predictions})

results.to_csv("submission.csv", index = False)