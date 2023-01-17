# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.gridspec as gridspec

from datetime import datetime

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import matplotlib.style as style

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
def missing_percentage(df):

    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""

    ## the two following line may seem complicated but its actually very simple. 

    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])



missing_percentage(train)

missing_percentage(test)
def plotting_3_chart(df, feature):

    ## Importing seaborn, matplotlab and scipy modules. 

    import seaborn as sns

    import matplotlib.pyplot as plt

    import matplotlib.gridspec as gridspec

    from scipy import stats

    import matplotlib.style as style

    style.use('fivethirtyeight')



    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(15,10))

    ## creating a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    #gs = fig3.add_gridspec(3, 3)



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

    

plotting_3_chart(train, 'SalePrice')
## Getting the correlation of all the features with target variable. 

(train.corr()**2)["SalePrice"].sort_values(ascending = False)[1:]
def customized_scatterplot(y, x):

        ## Sizing the plot. 

    style.use('fivethirtyeight')

    plt.subplots(figsize = (15,10))

    ## Plotting target variable with predictor variable(OverallQual)

    sns.scatterplot(y = y, x = x);
customized_scatterplot(train.SalePrice, train.OverallQual)
## Deleting those two values with outliers. 

train = train[train.GrLivArea < 4500]

train.reset_index(drop = True, inplace = True)



## save a copy of this dataset so that any changes later on can be compared side by side.

previous_train = train.copy()
## trainsforming target variable using numpy.log1p, 

train["SalePrice"] = np.log1p(train["SalePrice"])



## Plotting the newly transformed response variable

plotting_3_chart(train, 'SalePrice')
## Customizing grid for two plots. 

fig, (ax1, ax2) = plt.subplots(figsize = (20,6), ncols=2, sharey = False, sharex=False)

## doing the first scatter plot. 

sns.residplot(x = previous_train.GrLivArea, y = previous_train.SalePrice, ax = ax1)

## doing the scatter plot for GrLivArea and SalePrice. 

sns.residplot(x = train.GrLivArea, y = train.SalePrice, ax = ax2);
## Dropping the "Id" from train and test set. 

# train.drop(columns=['Id'],axis=1, inplace=True)



train.drop(columns=['Id'],axis=1, inplace=True)

test.drop(columns=['Id'],axis=1, inplace=True)



## Saving the target values in "y_train". 

y = train['SalePrice'].reset_index(drop=True)







# getting a copy of train

previous_train = train.copy()
## Combining train and test datasets together so that we can do all the work at once. 

all_data = pd.concat((train, test)).reset_index(drop = True)

## Dropping the target variable. 

all_data.drop(['SalePrice'], axis = 1, inplace = True)
missing_percentage(all_data)
## Some missing values are intentionally left blank, for example: In the Alley feature 

## there are blank values meaning that there are no alley's in that specific house. 

missing_val_col = ["Alley", 

                   "PoolQC", 

                   "MiscFeature",

                   "Fence",

                   "FireplaceQu",

                   "GarageType",

                   "GarageFinish",

                   "GarageQual",

                   "GarageCond",

                   'BsmtQual',

                   'BsmtCond',

                   'BsmtExposure',

                   'BsmtFinType1',

                   'BsmtFinType2',

                   'MasVnrType']



for i in missing_val_col:

    all_data[i] = all_data[i].fillna('None')
## the "OverallCond" and "OverallQual" of the house. 

# all_data['OverallCond'] = all_data['OverallCond'].astype(str) 

# all_data['OverallQual'] = all_data['OverallQual'].astype(str)



## Zoning class are given in numerical; therefore converted to categorical variables. 

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



## Important years and months that should be categorical variables not numerical. 

# all_data['YearBuilt'] = all_data['YearBuilt'].astype(str)

# all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype(str)

# all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str) 

all_data['Functional'] = all_data['Functional'].fillna('Typ') 

all_data['Utilities'] = all_data['Utilities'].fillna('AllPub') 

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0]) 

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA") 

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr")

## These features are continous variable, we used "0" to replace the null values. 

missing_val_col2 = ['BsmtFinSF1',

                    'BsmtFinSF2',

                    'BsmtUnfSF',

                    'TotalBsmtSF',

                    'BsmtFullBath', 

                    'BsmtHalfBath', 

                    'GarageYrBlt',

                    'GarageArea',

                    'GarageCars',

                    'MasVnrArea']



for i in missing_val_col2:

    all_data[i] = all_data[i].fillna(0)

    

## Replaced all missing values in LotFrontage by imputing the median value of each neighborhood. 

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))
missing_percentage(all_data)
sns.distplot(all_data['1stFlrSF']);
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)



skewed_feats
## Fixing Skewed features using boxcox transformation. 





def fixing_skewness(df):

    """

    This function takes in a dataframe and return fixed skewed dataframe

    """

    ## Import necessary modules 

    from scipy.stats import skew

    from scipy.special import boxcox1p

    from scipy.stats import boxcox_normmax

    

    ## Getting all the data that are not of "object" type. 

    numeric_feats = df.dtypes[df.dtypes != "object"].index



    # Check the skew of all numerical features

    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skewed_feats[abs(skewed_feats) > 0.5]

    skewed_features = high_skew.index



    for feat in skewed_features:

        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))



fixing_skewness(all_data)

sns.distplot(all_data['1stFlrSF']);
all_data = all_data.drop(['Utilities', 'Street', 'PoolQC',], axis=1)



# feture engineering a new feature "TotalFS"

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']



all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] +

                                 all_data['1stFlrSF'] + all_data['2ndFlrSF'])



all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +

                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))



all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +

                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +

                              all_data['WoodDeckSF'])
all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
all_data.shape
## Creating dummy variable 

final_features = pd.get_dummies(all_data).reset_index(drop=True)

final_features.shape
X = final_features.iloc[:len(y), :]



X_sub = final_features.iloc[len(y):, :]
outliers = [30, 88, 462, 631, 1322]

X = X.drop(X.index[outliers])

y = y.drop(y.index[outliers])
def overfit_reducer(df):

    """

    This function takes in a dataframe and returns a list of features that are overfitted.

    """

    overfit = []

    for i in df.columns:

        counts = df[i].value_counts()

        zeros = counts.iloc[0]

        if zeros / len(df) * 100 > 99.94:

            overfit.append(i)

    overfit = list(overfit)

    return overfit





overfitted_features = overfit_reducer(X)



X = X.drop(overfitted_features, axis=1)

X_sub = X_sub.drop(overfitted_features, axis=1)
X.shape,y.shape, X_sub.shape
## Train test s

from sklearn.model_selection import train_test_split

## Train test split follows this distinguished code pattern and helps creating train and test set to build machine learning. 

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .33, random_state = 0)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
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
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
print('START Fit')



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



# print('GradientBoosting')

# gbr_model_full_data = gbr.fit(X, y)



print('xgboost')

xgb_model_full_data = xgboost.fit(X, y)



print('lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)
def blend_models_predict(X):

    return ((0.1 * elastic_model_full_data.predict(X)) + \

            (0.05 * lasso_model_full_data.predict(X)) + \

            (0.2 * ridge_model_full_data.predict(X)) + \

            (0.1 * svr_model_full_data.predict(X)) + \

#             (0.1 * gbr_model_full_data.predict(X)) + \

            (0.15 * xgb_model_full_data.predict(X)) + \

            (0.1 * lgb_model_full_data.predict(X)) + \

            (0.3 * stack_gen_model.predict(np.array(X))))
print('RMSLE score on train data:')

print(rmsle(y, blend_models_predict(X)))
print('Predict submission')

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))
print('Blend with Top Kernels submissions\n')

sub_1 = pd.read_csv('../input/1-house-prices-solution-top-1/best_submission.csv')

sub_2 = pd.read_csv('../input/1-house-prices-solution-top-1/new_submission.csv')

sub_3 = pd.read_csv('../input/1-house-prices-solution-top-1/submission.csv')

submission.iloc[:,1] = np.floor((0.00 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 

                                (1.00 * sub_1.iloc[:,1]) + 

                                (0.00 * sub_2.iloc[:,1]) + 

                                (0.00 * sub_3.iloc[:,1]))
q1 = submission['SalePrice'].quantile(0.005)

q2 = submission['SalePrice'].quantile(0.995)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission0100.csv", index=False)
submission.iloc[:,1] = np.floor((0.00 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 

                                (0.00 * sub_1.iloc[:,1]) + 

                                (1.00 * sub_2.iloc[:,1]) + 

                                (0.00 * sub_3.iloc[:,1]))

q1 = submission['SalePrice'].quantile(0.005)

q2 = submission['SalePrice'].quantile(0.995)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission0010.csv", index=False)
#Failed

submission.iloc[:,1] = np.floor((0.00 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 

                                (0.00 * sub_1.iloc[:,1]) + 

                                (0.00 * sub_2.iloc[:,1]) + 

                                (1.00 * sub_3.iloc[:,1]))

q1 = submission['SalePrice'].quantile(0.005)

q2 = submission['SalePrice'].quantile(0.995)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission0001.csv", index=False)
submission.iloc[:,1] = np.floor((0.33 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 

                                (0.34 * sub_1.iloc[:,1]) + 

                                (0.33 * sub_2.iloc[:,1]))

q1 = submission['SalePrice'].quantile(0.005)

q2 = submission['SalePrice'].quantile(0.995)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission333433.csv", index=False)
submission.iloc[:,1] = np.floor((0.30 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 

                                (0.40 * sub_1.iloc[:,1]) + 

                                (0.30 * sub_2.iloc[:,1]))

q1 = submission['SalePrice'].quantile(0.005)

q2 = submission['SalePrice'].quantile(0.995)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission304030.csv", index=False)