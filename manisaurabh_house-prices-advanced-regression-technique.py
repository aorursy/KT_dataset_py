# Import the necessary libraries

import numpy as np

import pandas as pd

import os

import time

import warnings

import os

from six.moves import urllib

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

warnings.filterwarnings('ignore')

%matplotlib inline

plt.style.use('seaborn')

from scipy.stats import norm, skew
# Scalers

from sklearn.preprocessing import MinMaxScaler

from sklearn.utils import shuffle

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion



# Models



from sklearn.linear_model import Lasso, ElasticNet

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error



#regression

from sklearn.linear_model import LinearRegression,Ridge,LassoCV,RidgeCV

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor



from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix

from scipy.stats import reciprocal, uniform



# Cross-validation

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.model_selection import cross_validate



import xgboost as xgb

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



# GridSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



#Common data processors

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array

from scipy import sparse
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sample = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.shape
test.shape
#Drop the id column



train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
# It seems that the price of recent-built houses are higher



plt.figure(figsize=(15,8))

sns.boxplot(train.YearBuilt, train.SalePrice)



# From the graph we can surely see plenty of outliers.
plt.figure(figsize=(12,6))

plt.scatter(x=train.GrLivArea, y=train.SalePrice)

plt.xlabel("GrLivArea", fontsize=13)

plt.ylabel("SalePrice", fontsize=13)

plt.ylim(0,800000)
plt.figure(figsize=(12,6))

plt.scatter(x=train.OverallQual, y=train.SalePrice)

plt.xlabel("GrLivArea", fontsize=13)

plt.ylabel("SalePrice", fontsize=13)

plt.ylim(0,800000)
train.drop(train[(train['GrLivArea']>=4500) & (train['SalePrice']<300000)].index, inplace=True)

train.reset_index(drop=True, inplace=True)
# Graphs after removing outliers

plt.figure(figsize=(12,6))

plt.scatter(x=train.GrLivArea, y=train.SalePrice)

plt.xlabel("GrLivArea", fontsize=13)

plt.ylabel("SalePrice", fontsize=13)

plt.ylim(0,800000)
train.shape
train.info()
train.describe()
# Checking if the log is required for the housing sales price

plt.subplot(1, 2, 1)

sns.distplot(train.SalePrice, kde=True, fit = norm)
#Plot is right skewed, so we need to normalize this distribution



plt.subplot(1, 2, 2)

sns.distplot(np.log1p(train.SalePrice + 1), kde=True, fit = norm)

plt.xlabel('Log SalePrice')
#Applying log to house price

train.SalePrice = np.log1p(train.SalePrice)
train_y = train.SalePrice.reset_index(drop=True)

train_x = train.drop(['SalePrice'], axis=1)

test_x = test
train_x.shape
test_x.shape
total_features = pd.concat([train_x, test_x]).reset_index(drop=True)

total_features.shape
nulls = np.sum(total_features.isnull())

nullcols = nulls.loc[(nulls != 0)]

dtypes = total_features.dtypes

dtypes2 = dtypes.loc[(nulls != 0)]

info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)

print(info)

print("There are", len(nullcols), "columns with missing values")
total_features['Functional'] = total_features['Functional'].fillna('Typ')

total_features['Electrical'] = total_features['Electrical'].fillna("SBrkr")

total_features['KitchenQual'] = total_features['KitchenQual'].fillna("TA")



total_features['Exterior1st'] = total_features['Exterior1st'].fillna(total_features['Exterior1st'].mode()[0])

total_features['Exterior2nd'] = total_features['Exterior2nd'].fillna(total_features['Exterior2nd'].mode()[0])



total_features['SaleType'] = total_features['SaleType'].fillna(total_features['SaleType'].mode()[0])
pd.set_option('max_columns', None)

total_features[total_features['PoolArea'] > 0 & total_features['PoolQC'].isnull()]
total_features.loc[2418, 'PoolQC'] = 'Fa'

total_features.loc[2501, 'PoolQC'] = 'Gd'

total_features.loc[2597, 'PoolQC'] = 'Fa'
pd.set_option('max_columns', None)

total_features[(total_features['GarageType'] == 'Detchd') & total_features['GarageYrBlt'].isnull()]
total_features.loc[2124, 'GarageYrBlt'] = total_features['GarageYrBlt'].median()

total_features.loc[2574, 'GarageYrBlt'] = total_features['GarageYrBlt'].median()



total_features.loc[2124, 'GarageFinish'] = total_features['GarageFinish'].mode()[0]

total_features.loc[2574, 'GarageFinish'] = total_features['GarageFinish'].mode()[0]



total_features.loc[2574, 'GarageCars'] = total_features['GarageCars'].median()



total_features.loc[2124, 'GarageArea'] = total_features['GarageArea'].median()

total_features.loc[2574, 'GarageArea'] = total_features['GarageArea'].median()



total_features.loc[2124, 'GarageQual'] = total_features['GarageQual'].mode()[0]

total_features.loc[2574, 'GarageQual'] = total_features['GarageQual'].mode()[0]



total_features.loc[2124, 'GarageCond'] = total_features['GarageCond'].mode()[0]

total_features.loc[2574, 'GarageCond'] = total_features['GarageCond'].mode()[0]
# Basement Variables with NA, are now filled



total_features.loc[332, 'BsmtFinType2'] = 'ALQ' #since smaller than SF1

total_features.loc[947, 'BsmtExposure'] = 'No' 

total_features.loc[1485, 'BsmtExposure'] = 'No'

total_features.loc[2038, 'BsmtCond'] = 'TA'

total_features.loc[2183, 'BsmtCond'] = 'TA'

total_features.loc[2215, 'BsmtQual'] = 'Po' #v small basement so let's do Poor.

total_features.loc[2216, 'BsmtQual'] = 'Fa' #similar but a bit bigger.

total_features.loc[2346, 'BsmtExposure'] = 'No' #unfinished bsmt so prob not.

total_features.loc[2522, 'BsmtCond'] = 'Gd' #cause ALQ for bsmtfintype1
subclass_group = total_features.groupby('MSSubClass')

Zoning_modes = subclass_group['MSZoning'].apply(lambda x : x.mode()[0])

total_features['MSZoning'] = total_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
neighborhood_group = total_features.groupby('Neighborhood')

lot_medians = neighborhood_group['LotFrontage'].median()

total_features['LotFrontage'] = total_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
#Filling in the rest of the NA's



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in total_features.columns:

    if total_features[i].dtype in numeric_dtypes: 

        numerics.append(i)

        

total_features.update(total_features[numerics].fillna(0))



# remaining columns 



columns = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", 

         "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", 

         "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]



for col in columns:

    total_features.update(total_features[col].fillna("None", inplace=True))





nulls = np.sum(total_features.isnull())

nullcols = nulls.loc[(nulls != 0)]

dtypes = total_features.dtypes

dtypes2 = dtypes.loc[(nulls != 0)]

info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)

print(info)

print("There are", len(nullcols), "columns with missing values")
total_features = total_features.drop(['Utilities','Street'], axis=1)
#FEATURE ENGINEERING



total_features['Total_sqr_footage'] = (total_features['BsmtFinSF1'] + total_features['BsmtFinSF2'] +

                                 total_features['1stFlrSF'] + total_features['2ndFlrSF'])



total_features['Total_Bathrooms'] = (total_features['FullBath'] + (0.5*total_features['HalfBath']) + 

                               total_features['BsmtFullBath'] + (0.5*total_features['BsmtHalfBath']))



total_features['Total_porch_sf'] = (total_features['OpenPorchSF'] + total_features['3SsnPorch'] +

                              total_features['EnclosedPorch'] + total_features['ScreenPorch'] +

                             total_features['WoodDeckSF'])





#simplified features

total_features['haspool'] = total_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

total_features['has2ndfloor'] = total_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

total_features['hasgarage'] = total_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

total_features['hasbsmt'] = total_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

total_features['hasfireplace'] = total_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
total_features.shape
final_features = pd.get_dummies(total_features).reset_index(drop=True)

final_features.shape
final_train_x = final_features.iloc[:len(train_y),:]

final_test_x = final_features.iloc[len(final_train_x):,:] 
final_train_x.shape
final_test_x.shape
#Now let's use t-SNE to reduce dimensionality down to 2D so we can plot the dataset:



from sklearn.manifold import TSNE



tsne = TSNE(n_components=2, random_state=42, verbose = 2)

TSNE_X = tsne.fit_transform(final_train_x)

TSNE_X_test = tsne.fit_transform(final_test_x)
plt.figure(figsize=(13,10))

plt.scatter(TSNE_X[:, 0], TSNE_X[:, 1], c=train_y, cmap="jet")

plt.axis('off')

plt.colorbar()

plt.show()
from sklearn.decomposition import PCA



PCA_train_x = PCA(n_components=300, random_state=42).fit_transform(final_train_x)

plt.scatter(PCA_train_x[:, 0], PCA_train_x[:, 1], c=train_y, cmap="jet")

plt.axis('off')

plt.colorbar()

plt.show()
from sklearn.decomposition import KernelPCA



lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)

sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)





plt.figure(figsize=(11, 4))

for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), 

                            (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):

       

    plt.subplot(subplot)

    plt.title(title, fontsize=14)

    plt.scatter(PCA_train_x[:, 0], PCA_train_x[:, 1], c=train_y, cmap=plt.cm.hot)

    plt.xlabel("$z_1$", fontsize=18)

    if subplot == 131:

        plt.ylabel("$z_2$", fontsize=18, rotation=0)

    plt.grid(True)



plt.show()
from sklearn.manifold import LocallyLinearEmbedding



lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)

PCA_X = lle.fit_transform(final_train_x)
plt.title("Unrolled swiss roll using LLE", fontsize=14)

plt.scatter(PCA_X [:, 0], PCA_X [:, 1], c= train_y, cmap=plt.cm.hot)

plt.xlabel("$z_1$", fontsize=18)

plt.ylabel("$z_2$", fontsize=18)

plt.axis([-0.100, 0.215, -0.043, 0.14])

plt.grid(True)

plt.show()
pca_tsne = Pipeline([

    ("pca", PCA(n_components=0.95, random_state=42)),

    ("tsne", TSNE(n_components=2, random_state=42))

])

X_pca_tsne = pca_tsne.fit_transform(final_train_x)

plt.title("PCA and TSNE", fontsize=14)

plt.scatter(X_pca_tsne [:, 0], X_pca_tsne [:, 1], c= train_y, cmap=plt.cm.hot)

plt.xlabel("$z_1$", fontsize=18)

plt.ylabel("$z_2$", fontsize=18)

plt.show()
#Random Forest Regressor.

forest_class = RandomForestRegressor(random_state = 42)



n_estimators = [10,70,500,700]

max_features = ["auto",'sqrt','log2']



param_grid_forest = {'n_estimators' : n_estimators, 'max_features' : max_features}



rand_search_forest = RandomizedSearchCV(forest_class, param_grid_forest, cv = 4, 

                                        scoring='neg_mean_squared_error', n_jobs = -1, verbose=2)



rand_search_forest.fit(final_train_x, train_y)
random_estimator = rand_search_forest.best_estimator_ 

y_pred_rf= random_estimator.predict(final_train_x)

rf_msle = mean_squared_error(train_y, y_pred_rf)

rf_rmsle = np.sqrt(rf_msle)

rf_rmsle
GB_Regressor = GradientBoostingRegressor(random_state = 42)



n_estimators = [50,500]



param_grid_grad_boost_class = {'n_estimators' : n_estimators}



rand_search_grad_boost_class = GridSearchCV(GB_Regressor, param_grid_grad_boost_class, cv = 4, scoring='neg_mean_squared_error', 

                               refit = True, n_jobs = -1, verbose = 2)



rand_search_grad_boost_class.fit(final_train_x, train_y)
gb_estimator = rand_search_grad_boost_class.best_estimator_ 

y_pred_gb= gb_estimator.predict(final_train_x)

gb_msle = mean_squared_error(train_y, y_pred_gb)

gb_rmsle = np.sqrt(gb_msle)

gb_rmsle
en = ElasticNet()

en.fit(final_train_x, train_y)
#Implement an Elastic Net regressor



ElasticRegressor = ElasticNet()



alpha = [.0001,.0005,.005,.05,1]



param_grid_elastic = {'alpha' : alpha}



rand_search_elastic = GridSearchCV(ElasticRegressor, param_grid_elastic, cv = 4, scoring='neg_mean_squared_error', 

                               refit = True, n_jobs = -1, verbose = 2)



rand_search_elastic.fit(final_train_x, train_y)
elastic_estimator = rand_search_elastic.best_estimator_

y_pred_elastic = elastic_estimator.predict(final_train_x)

elastic_msle = mean_squared_error(train_y, y_pred_elastic)

elastic_rmsle = np.sqrt(elastic_msle)

elastic_rmsle
#Implement a lasso regresso



LassoRegressor = LassoCV()



max_iter = [50,100,500,1000]



param_grid_lasso = {'max_iter' : max_iter}



rand_search_lasso = GridSearchCV(LassoRegressor, param_grid_lasso, cv = 4, scoring='neg_mean_squared_error', 

                               refit = True, n_jobs = -1, verbose = 2)



rand_search_lasso.fit(final_train_x, train_y)
lasso_estimator = rand_search_lasso.best_estimator_

y_pred_lasso= lasso_estimator.predict(final_train_x)

lasso_msle = mean_squared_error(train_y, y_pred_lasso)

lasso_rmsle = np.sqrt(lasso_msle)

lasso_rmsle
xgb = XGBRegressor(learning_rate =0.01, n_estimators=3460, max_depth=3,

                     min_child_weight=0 ,gamma=0, subsample=0.7,

                     colsample_bytree=0.7,objective= 'reg:linear',

                     nthread=4,scale_pos_weight=1,seed=27, reg_alpha=0.00006)



xgb_fit = xgb.fit(final_train_x, train_y)
y_pred_xgb= xgb_fit.predict(final_train_x)

xgb_msle = mean_squared_error(train_y, y_pred_xgb)

xgb_rmsle = np.sqrt(xgb_msle)

xgb_rmsle
lgbm_model = LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



lgbm_fit = xgb.fit(final_train_x, train_y)
y_pred_lgbm= lgbm_fit.predict(final_train_x)

lgbm_msle = mean_squared_error(train_y, y_pred_lgbm)

lgbm_rmsle = np.sqrt(lgbm_msle)

lgbm_rmsle
y_pred_gb_test= gb_estimator.predict(final_test_x)

y_pred_rf_test= random_estimator.predict(final_test_x)

y_pred_las_test = lasso_estimator.predict(final_test_x)

y_pred_elas_test = elastic_estimator.predict(final_test_x)

y_pred_xgb_test = xgb_fit.predict(final_test_x)

y_pred_lgbm_test = lgbm_fit.predict(final_test_x)
submission = sample

submission.iloc[:,1] = (np.expm1(y_pred_gb_test) + np.expm1(y_pred_rf_test) + np.expm1(y_pred_xgb_test) + np.expm1(y_pred_lgbm_test))/ 4

submission.to_csv('submission.csv', index = False)