import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import xgboost as xgb

from sklearn import metrics

from scipy.stats import skew

import scipy.stats as stats

from scipy.special import boxcox1p

from sklearn import linear_model

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Lasso

from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import RandomForestRegressor

%matplotlib inline



import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.shape, test.shape
sns.scatterplot(x = train['GrLivArea'],y = train['SalePrice'])
train = train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index)
sns.scatterplot(y = train['SalePrice'], x = train['GrLivArea'])
sns.distplot(train['SalePrice'], fit = stats.norm)
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'], fit = stats.norm)
df_num = train.select_dtypes(include = np.number)

for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['SalePrice'])
train_corr = train

train_features = train.drop('SalePrice', axis = 1)

y = train.SalePrice.reset_index(drop=True)

test_features = test

features = pd.concat([train_features, test_features])
features.shape
features.head()
sns.set_style("whitegrid")

missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(ascending = False,inplace=True)

missing.plot.bar()
def missing_vals(df):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = (total/len(df)*100)

    missing = percent[percent>0]

    return pd.DataFrame( missing, columns = ['Percent'])

missing_vals(features)
# Dropping with a threshold of 75% missing values

features = features.dropna(thresh = len(features)*0.75, axis=1)



#Convert Year Variables to strings

for i in 'MSSubClass YrSold MoSold GarageYrBlt'.split():

    features[i] = features[i].apply(lambda x: str(x))

    

#LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood    

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



#Utilities Electrical, KitchenQual has one Na values and Functional has two Na values and MSZoning has 4. The Na values was therefore set to the Mode

for i in 'Utilities Functional Electrical KitchenQual MSZoning'.split():

    features[i] = features[i].fillna(features[i].mode()[0])

    

#GarageType, GarageFinish, GarageQual ,GarageCond are all categorical values, missing vals are replaced with None

for i in 'GarageType GarageFinish GarageQual GarageCond'.split():

    features[i] = features[i].fillna('None')

    

#The missing values are replaced with 0 because if GarageYrBlt, GarageArea, GarageCars, are 0 and numerical features it means that there is no garage

for i in 'GarageYrBlt GarageArea GarageCars'.split():

    features[i] = features[i].fillna(0)



#If any of the numerical Basement variables are zero, it means there is simply no Basement    

for i in 'BsmtFinSF1 BsmtFinSF2 BsmtUnfSF TotalBsmtSF BsmtFullBath BsmtHalfBath'.split():

    features[i] = features[i].fillna(0)



#If any of the categorical variables for Basement is zero, it means there is simply no Basement    

for i in 'BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinType2'.split():

    features[i] = features[i].fillna('None')



# If there is missing vals in the Masonary category, it simply means no masonary is part of the house, the categorical variable (MasVnrType) is set to None

# and the numerical variable (MasVnrArea) is set to 0

features["MasVnrType"] = features["MasVnrType"].fillna("None")

features["MasVnrArea"] = features["MasVnrArea"].fillna(0)



#Exterior1, Exterior2 and SaleType has 1 missing value, and because it is a categorical value, it will be set to the mode

for i in 'Exterior1st Exterior2nd SaleType'.split():

    features[i] = features[i].fillna(features[i].mode()[0])



#MSSubClass has one missing value, it will be set to None

features['MSSubClass'] = features['MSSubClass'].fillna("None")
missing_vals(features)
#features['YrBuiltandRemod'] = features['YearBuilt']+features['YearRemodAdd'] <-- Removed



features['TotSqrSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
feature_en = 'PoolArea Fireplaces GarageArea TotalBsmtSF 2ndFlrSF'.split()

for i in feature_en:

    features['Has'+i] = features[i].apply(lambda x: 1 if x> 0 else 0)
numeric_feats = features.dtypes[features.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = features[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewed_features = skewness.index

for feat in skewed_features:

    features[feat] = boxcox1p(features[feat],0.5) #0.5 at the Power Parameter(lmbda) gave the best results
features_cat = features.copy()
features = pd.get_dummies(features, drop_first = True)

features.shape
train_set=features[:len(y)]

test_set=features[len(train_set):]

train_set.head()
# Drop the Id column

train_set.drop('Id', axis=1, inplace=True, errors='raise')

test_set.drop('Id', axis=1, inplace=True, errors='raise')
X = train_set

test_X = test_set
# Set up variables

X_train = X

X_test = test_X

y_train = y



def rmse(model):

    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 10))

    return(rmse)
#Lasso

model_lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0003))



model_lasso.fit(X_train, y_train)



lasso_rmse = rmse(model_lasso).mean()



alphas = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]



# Creating Plots to visually show the best alpha to choose

lasso_alpha = []

for i in alphas:

    lasso_alpha.append(rmse(make_pipeline(RobustScaler(), Lasso(alpha = i))).mean())





lasso_alpha = pd.Series(lasso_alpha, index = alphas)

lasso_alpha.plot(title = "Alpha Validation")

plt.xlabel("Alpha")

plt.ylabel("Rmse")
#RandomForest



model_random_forest = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators = 100 , oob_score = True, n_jobs = -1))



model_random_forest.fit(X_train, y_train)



randomforest_rmse = rmse(model_random_forest).mean()
#Ridge



ridge = make_pipeline(RobustScaler(), linear_model.Ridge())



ridge.fit(X_train, y_train)



ridge_rmse = rmse(ridge).mean()
# XgBoost



xg_boost = make_pipeline(RobustScaler(), xgb.XGBRegressor())



xg_boost.fit(X_train, y_train)



xgboost_rmse = rmse(xg_boost).mean()
results_df = pd.DataFrame({'Model': ['Lasso', 'RandomForest', 'Ridge', 'XGBoost'], 'Score': [lasso_rmse, randomforest_rmse,ridge_rmse, xgboost_rmse]})

results_df = results_df.sort_values(by='Score', ascending=True).reset_index(drop=True)

results_df
df = pd.DataFrame({'Id': test.Id, 'SalePrice':np.exp(model_lasso.predict(X_test))})

df.to_csv('Submission_final.csv', index = False)
corr_map = train_corr.corr()

sns.heatmap(corr_map)
li_cat_feats = list(train_corr.select_dtypes(include = 'object'))



fig, axs = plt.subplots(nrows = 15, ncols = 3, figsize=(12,45))



for r in range(0,15):

    for c in range(0,3):  

        i = r*3+c

        if i < len(li_cat_feats):

            sns.boxplot(x=li_cat_feats[i], y=train_corr['SalePrice'], data=train_corr, ax = axs[r][c])

            

    

plt.tight_layout()    

plt.show() 



# This graph was the best way to display categorical variables and the variance it had with the SalePrice

# And we got it from https://www.kaggle.com/dejavu23/house-prices-eda-to-ml-beginner
cat_high_corr = ['HeatingQC','GarageType','MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'KitchenQual', 'SaleType', 'SaleType']
num_features = train_corr.select_dtypes(include = np.number)
num_corr = num_features.corr()['SalePrice'][:-1] # -1 because the latest row is SalePrice

golden_features_list = num_corr[abs(num_corr) > 0.3].sort_values(ascending=False)

golden_features_list
features = features_cat[cat_high_corr + list(golden_features_list.index)]
features_cat = pd.get_dummies(features, drop_first = True)
features_cat.shape
train_set=features_cat[:len(y)]

test_set=features_cat[len(train_set):]

train_set.head()
X = train_set

test_X = test_set
# Set up variables

X_train = X

X_test = test_X

y_train = y
#Lasso



model_lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0002))



model_lasso.fit(X_train, y_train)



lasso_rmse = rmse(model_lasso).mean()



alphas = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]



# Creating Plots to visually show the best alpha to choose

lasso_alpha = []

for i in alphas:

    lasso_alpha.append(rmse(make_pipeline(RobustScaler(), Lasso(alpha = i))).mean())





lasso_alpha = pd.Series(lasso_alpha, index = alphas)

lasso_alpha.plot(title = "Alpha Validation")

plt.xlabel("Alpha")

plt.ylabel("Rmse")
#RandomForest



model_random_forest = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators = 200 , oob_score = True, n_jobs = -1))



model_random_forest.fit(X_train, y_train)



randomforest_rmse = rmse(model_random_forest).mean()
#Ridge



ridge = make_pipeline(RobustScaler(), linear_model.Ridge())



ridge.fit(X_train, y_train)



ridge_rmse = rmse(ridge).mean()
# XgBoost



xg_boost = make_pipeline(RobustScaler(), xgb.XGBRegressor())



xg_boost.fit(X_train, y_train)



xgboost_rmse = rmse(xg_boost).mean()
results_df = pd.DataFrame({'Model': ['Lasso', 'RandomForest', 'Ridge', 'XGBoost'], 'Score': [lasso_rmse, randomforest_rmse,ridge_rmse, xgboost_rmse]})

results_df = results_df.sort_values(by='Score', ascending=True).reset_index(drop=True)

results_df