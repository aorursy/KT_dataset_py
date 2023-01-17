# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #Data Viz

import matplotlib.pyplot as plt

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from scipy import stats

from datetime import datetime

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

import plotly.graph_objs as go

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import scipy.stats as stats

import sklearn.linear_model as linear_model

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor



import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing the data

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#Checking the structure of train and test datasets

print('Train shape: ', train.shape)

print('Test Shape: ', test.shape)



#Types of fields present in the test and train

print('Train: \n', train.dtypes.value_counts())

print('Test: \n', test.dtypes.value_counts())
#merging test and train

merged_df = pd.concat([train, test], axis = 0, sort = True)

print('Merged Shape: \n', merged_df.shape)

print('Merged : \n', merged_df.dtypes.value_counts())

#Extracting quantitative and qualitative fields

numeric_merged = merged_df.select_dtypes(include = ['int64','float64'])

obj_merged = merged_df.select_dtypes(include = ['object'])



#Shapes

print('Shape of Numeric: \n', numeric_merged.shape)

print('Shape of Object type: \n', numeric_merged.shape)
#Histograms for the numeric fields. We will create a 19X2 area for the 38 histograms.

fig = plt.figure()

for i, variable in enumerate (numeric_merged.columns):

    ax = fig.add_subplot(19, 2, i+1)

    numeric_merged[variable].hist(bins = 50, ax = ax, color = 'blue', alpha = 0.5, figsize = (50,150))

    ax.set_title (variable, fontsize = 35)

    ax.tick_params(axis = 'both', which = 'major',labelsize = 30)

    ax.tick_params(axis = 'both', which = 'minor',labelsize = 30)

    ax.set_xlabel('')

    

plt.show()

#Converting some numeric fields to object (categorical) fields

merged_df.loc[:, ['MSSubClass', 'OverallQual', 'OverallCond']] = merged_df.loc[:, ['MSSubClass', 'OverallQual', 'OverallCond']].astype('object')



#new shape of merged_df

print('New Shape of merged data: \n', merged_df.shape)



#Datatypes in merged_df post conversion to object

print('Merged structure : \n', merged_df.dtypes.value_counts())
#Generation & Visualisation of Correlation Matrix for train

c_mat = train.corr()

f, ax = plt.subplots(figsize=(15, 12))

sns.heatmap(c_mat, vmax=.8, square=True);
#Finding the 10 highest correlated fields to 'SalePrice'

n = 10 #Number of fields

cols = c_mat.nlargest(n, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#Scatter plot of GrLivArea v SalePrice

train.plot.scatter(x='GrLivArea', y='SalePrice', title = 'GrLivArea v SalePrice');
#Dropping GrLivArea > 4000

train.drop(train[train.GrLivArea>4000].index, inplace = True)

train.reset_index(drop = True, inplace = True)



display(train.shape)
#Scatter plot of GarageCars v SalePrice

train.plot.scatter(x='GarageCars', y='SalePrice', title = 'GarageCars v SalePrice');
#Scatter plot of YearBuilt v SalePrice

train.plot.scatter(x='YearBuilt', y='SalePrice', title = 'YearBuilt v SalePrice');



#Drop observations where YearBulit is less than 1893 sq.ft

train.drop(train[train.YearBuilt<1900].index, inplace = True)

train.reset_index(drop = True, inplace = True)
#Scatter plot of YearBuilt v SalePrice

train.plot.scatter(x='YearBuilt', y='SalePrice', title = 'YearBuilt v SalePrice');
#Scatter plot of OverallQual v SalePrice

train.plot.scatter(x='OverallQual', y='SalePrice', title = 'OverallQual v SalePrice');
#Scatter plot of FullBath v SalePrice

train.plot.scatter(x='FullBath', y='SalePrice', title = 'FullBath v SalePrice');



#Scatter plot of TotalBsmtSF v SalePrice

train.plot.scatter(x='TotalBsmtSF', y='SalePrice', title = 'TotalBsmtSF v SalePrice');
#Removing obs with 'TotalBsmtSF' > 3000

train.drop(train[train.TotalBsmtSF>3000].index, inplace = True)

train.reset_index(drop = True, inplace = True)
#Scatter plot of TotalBsmtSF v SalePrice after outlier removal

train.plot.scatter(x='TotalBsmtSF', y='SalePrice', title = 'TotalBsmtSF v SalePrice');
#Extracting 'SalePrice' and dropping from feature set

y_train = train.SalePrice

train.drop('SalePrice', axis = 1, inplace = True)

train.shape



#merging with test

merged_df = pd.concat([train, test], axis = 0)

merged_df.shape

#Converting some numeric fields to object (categorical) fields

merged_df.loc[:, ['MSSubClass', 'OverallQual', 'OverallCond']] = merged_df.loc[:, ['MSSubClass', 'OverallQual', 'OverallCond']].astype('object')



#new shape of merged_df

print('New Shape of merged data: \n', merged_df.shape)



#Datatypes in merged_df post conversion to object

print('Merged structure : \n', merged_df.dtypes.value_counts())
#Columns with missing information

missing_col = merged_df.columns[merged_df.isnull().any()].values

print(missing_col)
missing_col = len(merged_df) - merged_df.loc[:, np.sum(merged_df.isnull())>0].count()

missing_col
'''We can see that for 'MiscFeature', NaN means absence of the feature as per the description'''

merged_df['MiscFeature'].head(10)
'''We can see that for 'Fence', NaN means absence of the feature as per the description'''

merged_df['Fence'].head(10)
#Replacing the NaN in the above 14 fields fith "None"

replace_none = merged_df.loc[:, ['PoolQC','Fence','MiscFeature','GarageCond','GarageQual','GarageFinish','GarageType','Alley','FireplaceQu','BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual']]



for i in replace_none.columns:

    merged_df[i].fillna('None', inplace = True)

    

#After removal of the 'NaN'

missing_col = len(merged_df) - merged_df.loc[:, np.sum(merged_df.isnull())>0].count()

missing_col
#Replacing non-numeric missing by corresponding mode

replace_mode = merged_df.loc[:, ['MSZoning', 'Electrical', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'KitchenQual', 'Functional', 'SaleType']]



for i in replace_mode.columns:

    merged_df[i].fillna(merged_df[i].mode()[0], inplace = True)

    

#After removal of the 'NaN'

missing_col = len(merged_df) - merged_df.loc[:, np.sum(merged_df.isnull())>0].count()

missing_col
#Replacing numeric missing by corresponding Median

replace_mode = merged_df.loc[:, ['LotFrontage','MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']]



for i in replace_mode.columns:

    merged_df[i].fillna(merged_df[i].median(), inplace = True)

    

#After removal of the 'NaN'

missing_col = len(merged_df) - merged_df.loc[:, np.sum(merged_df.isnull())>0].count()

missing_col
#Checking Skewness of 'SalePrice'

print('Skewness of SalePrice is: ', y_train.skew())



#Checking normality of 'SalePrice'

sns.distplot(y_train, fit = norm)

figure = plt.figure()
#Taking log of 'SalePrice'

y_train = np.log1p(y_train)



#Re-checking normality after transformation

sns.distplot(y_train, fit = norm)

figure = plt.figure()
#Checking skewness of numeric features

numeric_skew = pd.DataFrame(data = merged_df.select_dtypes(include = ['int64', 'float64']).skew(), columns = ['Skewness'])

numeric_skew_sort = numeric_skew.sort_values(ascending = False, by = 'Skewness')

numeric_skew_sort
#Extract all numeric fields from merged_df

merged_df_num = merged_df.select_dtypes(include = ['int64', 'float64'])



#Transform the numeric fields with Skewness > 0.5

merged_df_skew = np.log1p(merged_df_num[merged_df_num.skew()[merged_df_num.skew() > 0.5].index])



#Pick up non-skewed features

merged_df_noskew = merged_df_num[merged_df_num.skew()[merged_df_num.skew() < 0.5].index]



#Merging Skewed and non-skewed population

merged_df_num_final = pd.concat([merged_df_skew, merged_df_noskew], axis = 1)



#Merge with the main numeric dataframe

merged_df_num.update(merged_df_num_final)
#Using StandardScaler to scale the numeric features

standard_scaler = StandardScaler()

merged_df_num_scaled = standard_scaler.fit_transform(merged_df_num)

merged_df_num_scaled = pd.DataFrame(data = merged_df_num_scaled, columns = merged_df_num.columns, index = merged_df_num.index)
#Extracting the categoricals

merged_df_cat = merged_df.select_dtypes(include = ['object']).astype('category')

merged_df_cat.shape



#Using LabelEncoder for the categorucal features

merged_df_cat_encoded = merged_df_cat.apply(LabelEncoder().fit_transform)

merged_df_cat_encoded.shape



#OneHotEncoding using Pandas

merged_df_one_hot = merged_df_cat.select_dtypes(include = ['category'])

merged_df_one_hot = pd.get_dummies(merged_df_one_hot, drop_first = True)

merged_df_one_hot.shape



#Merging label encoded and one hot encoded features together

merged_df_cat_encoded_final = pd.concat([merged_df_one_hot, merged_df_cat_encoded], axis = 1)



#Joining the processed numeric and categorical fields

merged_df_final_processed =pd.concat([merged_df_num_scaled, merged_df_cat_encoded_final], axis = 1)

merged_df_final_processed.shape
#Isolating Train and Test features

train_final = merged_df_final_processed.iloc[0:1438, :]

test_final = merged_df_final_processed.iloc[1438:, :]



#Target field

y_train = y_train



X = train_final

y = y_train
#Setting Random State

state = 40



#KFolds Cross Validation

kfolds = KFold(n_splits=10, shuffle=True, random_state=state)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
ridge = RidgeCV(alphas=alphas_alt, cv=kfolds)

lasso = LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds)

elasticnet = ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio)                            

svr = SVR(C= 20, epsilon= 0.008, gamma=0.0003)
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)                             
lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
'''ab = AdaBoostRegressor(random_state = state)

rf = RandomForestRegressor(n_jobs = -1, random_state = state)

knn = KNeighborsRegressor(n_jobs= -1)

dt = DecisionTreeRegressor(random_state = state)'''
score = cv_rmse(ridge)

score = cv_rmse(lasso)

print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(elasticnet)

print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(svr)

print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(lightgbm)

print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(gbr)

print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(xgboost)

print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



'''Excluding as RMSE was quite high'''



'''score = cv_rmse(ab)

print("adaboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(rf)

print("Random Forest: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(knn)

print("KNN: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )



score = cv_rmse(dt)

print("Decision Tree: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )'''
#Fitting the data



print('stack_gen')

stack_gen_model = stack_gen.fit(np.array(X), np.array(y))



print('elasticnet')

elastic_model_full_data = elasticnet.fit(X, y)



print('Lasso')

lasso_model_full_data = lasso.fit(X, y)



print('Ridge')

ridge_model_full_data = ridge.fit(X, y)



print('Svr')

svr_model_full_data = svr.fit(X, y)



print('GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)



print('xgboost')

xgb_model_full_data = xgboost.fit(X, y)



print('lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)
def blend_models_predict(X):

    return ((0.1 * elastic_model_full_data.predict(X)) + \

            (0.05 * lasso_model_full_data.predict(X)) + \

            (0.1 * ridge_model_full_data.predict(X)) + \

            (0.1 * svr_model_full_data.predict(X)) + \

            (0.1 * gbr_model_full_data.predict(X)) + \

            (0.15 * xgb_model_full_data.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.3 * stack_gen_model.predict(np.array(X))))
print('RMSLE score on train data:')

print(rmsle(y, blend_models_predict(X)))
#Creating Submission

print('Predict submission')

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(test_final)))
submission.head(5)

submission.to_csv('submission.csv',index=False)