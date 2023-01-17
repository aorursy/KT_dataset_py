%matplotlib inline

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb

from scipy import stats
from scipy.stats import norm, skew
from sklearn.linear_model import RidgeCV, LassoCV, LassoLarsCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, learning_curve, GridSearchCV
from scipy.stats.stats import pearsonr

warnings.filterwarnings("ignore")
# Loading CSV files

df_houses_train = pd.read_csv('../input/train.csv')
df_houses_test = pd.read_csv('../input/test.csv')

# Keeping the Id's

id_train = df_houses_train['Id']
id_test = df_houses_test['Id']

# Dropping the Id column because is not useful to make predictions

df_houses_train.drop(['Id'], axis=1, inplace=True)
df_houses_test.drop(['Id'], axis=1, inplace=True)
# See the correlation among the features
# When dealing with many features, let's first start with top-correlated features
corr = df_houses_train.corr()
top_corr = corr.index[abs(corr["SalePrice"]) > 0.45]

f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(df_houses_train[top_corr].corr(), 
            annot=True, 
            linewidths=.4, 
            fmt='.3f', 
            robust=True,
            ax=ax, 
            cmap='Reds',
            square=True);
# Removing some columns
# Alternative to df.drop(['column'],axis=1,inplace=True)

del df_houses_train['1stFlrSF']
del df_houses_train['TotRmsAbvGrd']
del df_houses_train['GarageArea']
del df_houses_train['GarageYrBlt']

del df_houses_test['1stFlrSF']
del df_houses_test['TotRmsAbvGrd']
del df_houses_test['GarageArea']
del df_houses_test['GarageYrBlt']
#My favorite plot: pairplot => focus on the first row 
sns.set()
columns = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'OverallQual', 'GarageCars', 'FullBath', 'YearBuilt']
sns.pairplot(df_houses_train[columns]);
# GrLivArea

plt.subplots(figsize=(12, 5))

plt.subplot(1, 2, 1)
g = sns.regplot(x = df_houses_train['GrLivArea'], 
                y = df_houses_train['SalePrice'], 
                fit_reg=True).set_title("With outliers")

print('Correlation (with outliers)   :', 
      pearsonr(df_houses_train['SalePrice'], df_houses_train['GrLivArea'])[0])

# Deleting outliers
# Simple method to eliminate outliers
condition = ((df_houses_train['GrLivArea']>4000) & (df_houses_train['SalePrice']<300000))
df_houses_train = df_houses_train.drop(df_houses_train[condition].index).reset_index(drop=True)

plt.subplot(1, 2, 2)
g = sns.regplot(x = df_houses_train['GrLivArea'], 
                y = df_houses_train['SalePrice'], 
                fit_reg=True).set_title("With no outliers")

print('Correlation (with no outliers):', 
      pearsonr(df_houses_train['SalePrice'], df_houses_train['GrLivArea'])[0])
# TotalBsmtSF

plt.subplots(figsize=(12, 5))

plt.subplot(1, 2, 1)
g = sns.regplot(x = df_houses_train['TotalBsmtSF'], 
                y = df_houses_train['SalePrice'], 
                fit_reg=True).set_title("With outliers")

print('Correlation (with outliers)   :', 
      pearsonr(df_houses_train['SalePrice'], df_houses_train['TotalBsmtSF'])[0])

# Deleting outliers

#condition = df_houses_train['TotalBsmtSF']>3000
condition = ((df_houses_train['TotalBsmtSF'] > 3000) & (df_houses_train['SalePrice'] < 300000))
df_houses_train = df_houses_train.drop(df_houses_train[condition].index).reset_index(drop=True)

plt.subplot(1, 2, 2)
g = sns.regplot(x = df_houses_train['TotalBsmtSF'], 
                y = df_houses_train['SalePrice'], 
                fit_reg=True).set_title("With no outliers")

print('Correlation (with no outliers):', 
      pearsonr(df_houses_train['SalePrice'], df_houses_train['TotalBsmtSF'])[0])
def get_avg_totalbsmt(col):
        mean = df_houses_train['TotalBsmtSF'].mean()
        return col if col != 0 else mean
# OverallQual

plt.subplots(figsize=(12, 5))

plt.subplot(1, 2, 1)
g = sns.regplot(x = df_houses_train['OverallQual'], 
                y = df_houses_train['SalePrice'], 
                fit_reg=True).set_title("With outliers")

print('Correlation (with outliers)   :', 
      pearsonr(df_houses_train['SalePrice'], df_houses_train['OverallQual'])[0])

# Deleting outliers

condition = ((df_houses_train['OverallQual']==10) & (df_houses_train['SalePrice']>650000))
df_houses_train = df_houses_train.drop(df_houses_train[condition].index).reset_index(drop=True)

plt.subplot(1, 2, 2)
g = sns.regplot(x = df_houses_train['OverallQual'], 
                y = df_houses_train['SalePrice'], 
                fit_reg=True).set_title("With no outliers")

print('Correlation (with no outliers):', 
      pearsonr(df_houses_train['SalePrice'], df_houses_train['OverallQual'])[0])
# View the distribution of SalePrice

fig, ax = plt.subplots(figsize=(8,5))
g = sns.distplot(df_houses_train["SalePrice"], 
                 color="g", 
                 label="Skewness : %.3f"%(df_houses_train["SalePrice"].skew()), 
                 hist_kws=dict(edgecolor="w", linewidth=1), ax=ax)
                 
g = g.legend(loc="best")
# Using log1p to get a better distribution

df_houses_train["SalePrice"] = np.log1p(df_houses_train["SalePrice"])
# View the distribution of SalePrice after applying Log

fig, ax = plt.subplots(figsize=(8,5))
g = sns.distplot(df_houses_train["SalePrice"], 
                 color="g", 
                 label="Skewness : %.3f"%(df_houses_train["SalePrice"].skew()), 
                 hist_kws=dict(edgecolor="w", linewidth=1),
                 ax=ax)
                 
g = g.legend(loc="best")
# Joining both datasets

train_size = len(df_houses_train)
y_train = df_houses_train['SalePrice']

all_houses =  pd.concat((df_houses_train, df_houses_test)).reset_index(drop=True)
all_houses.drop(['SalePrice'], axis=1, inplace=True)

# Filling empty values with NaN and checking the null values

all_houses.fillna(np.nan, inplace=True)
all_houses.head()
# Normalising a column of values between 0 and 1

def normal_minmax(col):
    return (col - col.min())/(col.max() - col.min())
# Amount of null values

all_houses.isnull().sum().sort_values(ascending=False).head(35)
columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
           'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
           'MasVnrType',
           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for col in columns:
    all_houses[col] = all_houses[col].fillna('None')
columns = ['GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
           'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

for col in columns:
    all_houses[col] = all_houses[col].fillna(0)
df_houses_train['TotalBsmtSF_AVG'] = df_houses_train['TotalBsmtSF'].apply(get_avg_totalbsmt)
columns = ['MSZoning', 'Functional', 'Utilities', 'Exterior1st', 'Exterior2nd',
           'KitchenQual', 'SaleType', 'Electrical']

for col in columns:
    print('Column [ {0} ] - Mode: {1}'.format(col, all_houses[col].mode()[0])) # Mode of the column
    all_houses[col] = all_houses[col].fillna(all_houses[col].mode()[0])
# LotFrontage: Linear feet of street connected to property

all_houses['LotFrontage'] = all_houses.groupby('Neighborhood')['LotFrontage'].transform(
    lambda lotFrontage: lotFrontage.fillna(lotFrontage.median())) # Median of the neighborhood
# Replacing name of the month by corresponding number

all_houses = all_houses.replace({'MoSold': {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                                            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov',12: 'Dec'}})
# Converting YrSold to string

all_houses['YrSold'] = all_houses['YrSold'].astype(str)
# Replacing MSSubClass by a combinated name

for subclass in all_houses['MSSubClass'].unique():
    all_houses = all_houses.replace({'MSSubClass': {subclass : 'SubClass_' + str(subclass)}})
# Creating categories using ordered features

cat_features = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual',
                'ExterCond', 'ExterQual', 'Fence', 'FireplaceQu', 'Functional', 'GarageCond', 
                'GarageFinish', 'GarageQual', 'HeatingQC', 'KitchenQual', 'PavedDrive', 'PoolQC', 'Utilities']

for feature in cat_features:
    all_houses[feature] = all_houses[feature].astype("category")
    
all_houses = all_houses.replace({
"BsmtExposure" : {"None" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
"BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
"BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
"Fence" : {"None" : 0, "MnWw" : 1, "GdWo": 2, "MnPrv" : 3, "GdPrv" : 4},
"Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},
"GarageFinish" : {"None" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},
"PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
"Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}})

all_houses['CentralAir'].replace({'Y':1,'N':0}, inplace=True)
columns = ['ExterQual','ExterCond','BsmtQual','BsmtCond', 'HeatingQC','KitchenQual',
           'FireplaceQu','GarageQual', 'GarageCond','PoolQC']
values = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0}

for col in columns:
    all_houses[col].replace(values, inplace=True)  
# Amount of numeric and string features

string_features = all_houses.select_dtypes(include=['object']).columns
numeric_features = all_houses.select_dtypes(include=['int64', 'float64']).columns
# Normalizing the numeric data

all_houses[numeric_features] = all_houses[numeric_features].apply(normal_minmax)
skewed_features = all_houses[numeric_features].apply(lambda x: skew(x))
skewed_features = skewed_features[abs(skewed_features) > 0.75]
skewed_features = skewed_features.index
skewed_features
# Treating the skewed features

all_houses[skewed_features] = np.log1p(all_houses[skewed_features])
all_houses = pd.get_dummies(all_houses, columns=string_features, drop_first=True)
df_houses_train = all_houses[:train_size]
df_houses_test = all_houses[train_size:]
X_train = df_houses_train
X_test = df_houses_test

print('Train:', X_train.shape)
print('Test :', X_test.shape)
# Validation of the model with Kfold stratified splitting the data into n_splits parts

seed = 42
n_folds = 5
kfold = KFold(n_folds, shuffle=True, random_state=seed).get_n_splits(X_train)
def get_rmse(model, x, y):
    rmse = np.sqrt(-cross_val_score(model, 
                                    x, 
                                    y=y, 
                                    scoring="neg_mean_squared_error", 
                                    cv=kfold))
    return rmse
def get_results(models, algorithms, X, y):
    # Getting all results from n_splits validations for each classifier

    clf_results = []
    for clf in models:
        clf_results.append(get_rmse(clf, X, y))

    # Getting the mean and standard deviation from each classifier's result after 5 validations

    clf_means = []
    clf_std = []
    for clf_result in clf_results:
        clf_means.append(clf_result.mean())
        clf_std.append(clf_result.std())

    # Let's see the best scores of each algorithm

    df_result = pd.DataFrame({"Means":clf_means, 
                              "Stds": clf_std, 
                              "Algorithm": algorithms})
    
    return df_result.sort_values(by=['Means'], ascending=True)
ridgeCV      = RidgeCV(alphas=np.arange(0.001, 0.1, 0.01))
lassoCV      = LassoCV(alphas=np.arange(0.001, 0.1, 0.01), eps=np.arange(1e-7, 1e-5, 1e-6))
lassoLarsCV  = LassoLarsCV(max_iter=1000, eps=1e-7)
elasticNetCV = ElasticNetCV(eps=1e-15)

gBoost  = GradientBoostingRegressor(random_state=seed)
xgbReg  = xgb.XGBRegressor(random_state=seed)
lgbmReg = lgb.LGBMRegressor()

# Dictionary of classifiers

clfs = dict()

clfs['RidgeCV'] = ridgeCV
clfs['LassoCV'] = lassoCV
clfs['LassoLarsCV'] = lassoLarsCV
clfs['ElasticNetCV'] = elasticNetCV
clfs['GBoosting'] = gBoost
clfs['XGBRegressor'] = xgbReg
clfs['LGBMRegressor'] = lgbmReg
def tuning(model, param_grid, X, y):
    
    grid_result = GridSearchCV(model,
                          param_grid=param_grid, 
                          cv=kfold, 
                          scoring="neg_mean_squared_error", 
                          n_jobs=-1, 
                          verbose=1)

    grid_result.fit(X, y)
    model_best = grid_result.best_estimator_

    # Best score
    print('Best score:', np.round(grid_result.best_score_, 4))

    # Best estimator
    print('Best estimator:', model_best)
    
    return model_best
## Search grid for optimal parameters (GradientBoostingRegressor)

parameters = {'n_estimators':[1800, 1900], 
            'learning_rate': [0.05], 
            'max_depth': [2, 3],
            'min_samples_leaf': [14, 16], 
            'max_features': ['sqrt'],
            'min_samples_split': [5, 6],
            'loss': ['huber', 'ls']}

gBoost_best = tuning(clfs['GBoosting'], parameters, X_train, y_train)
## Search grid for optimal parameters (XGBRegressor)

parameters = {'learning_rate': [0.01, 0.04], 
              'max_depth': [2, 3],
              'min_child_weight': [2, 3],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [1300, 1500]}

xgbReg_best = tuning(clfs['XGBRegressor'], parameters, X_train, y_train)
## Search grid for optimal parameters (LGBMRegressor)

parameters = {'objective': ['regression'],
              'num_leaves': [4, 5],
              'max_depth': [3, 4],
              'bagging_freq': [5, 6],
              'bagging_fraction': [0.8, 0.9],
              'learning_rate': [0.01, 0.05],
              'n_estimators': [700, 1000]}

lgbmReg_best = tuning(clfs['LGBMRegressor'], parameters, X_train, y_train)
clfs['GBoostingBest'] = gBoost_best
clfs['XGBRegressorBest'] = xgbReg_best
clfs['LGBMRegressorBest'] = lgbmReg_best

# Get the results

get_results(list(clfs.values()), list(clfs.keys()), X_train, y_train)
# Get the average score from some models

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
avg_models1 = AveragingModels(models = (lassoLarsCV, elasticNetCV, lassoCV, 
                                        ridgeCV, gBoost_best, xgbReg_best, lgbmReg_best))

avg_models2 = AveragingModels(models = (lassoLarsCV, elasticNetCV, xgbReg_best, gBoost_best))

avg_models3 = AveragingModels(models = (lassoLarsCV, elasticNetCV))
clfs['AVG Models 1'] = avg_models1
clfs['AVG Models 2'] = avg_models2
clfs['AVG Models 3'] = avg_models3

# Get the results

get_results([clfs[key] for key in list(clfs.keys()) if 'AVG' in key], 
            [key for key in list(clfs.keys()) if 'AVG' in key],
            X_train, y_train)
# Trainning model LassoLarsCV

clfs['LassoLarsCV'].fit(X_train, y_train)
# Printing the best features used to increase the price and 
# the ones didn't help for predicting the price (zero coefficient)

coefs = pd.Series(clfs['LassoLarsCV'].coef_,index=X_train.columns)

print('---------------------------------------')
print(sum(coefs==0),'Zero coefficients')
print(sum(coefs!=0),'Non-zero coefficients')
print('---------------------------------------')
print('Top 20 that increased price:')
print('---------------------------------------')
print(coefs.sort_values(ascending=False).head(20))
print('---------------------------------------')
print('Not affected the price:')
print('---------------------------------------')
print(coefs[coefs==0].index.tolist())
# Removing columns with zero coefficient

X_train_new = X_train.copy()
X_test_new = X_test.copy()
for column in X_train.columns:
    if column in coefs[coefs==0].index:
        del X_train_new[column]
        del X_test_new[column]
# Creating new features using the 20 best contributers

for col1 in coefs.sort_values(ascending=False).head(20).index:
    for col2 in coefs.sort_values(ascending=False).head(20).index:
        new_col = col1 + '_' + col2
        if new_col not in X_train_new.columns:
            X_train_new[new_col] = X_train_new[col1] * X_train_new[col2]
            X_test_new[new_col] = X_test_new[col1] * X_test_new[col2]
            
for col in coefs.sort_values(ascending=False).head(20).index:
    new_col = col + '_cubic'
    X_train_new[new_col] = X_train_new[col] ** 3
    X_test_new[new_col] = X_test_new[col] ** 3
# Get the results

kfold = KFold(n_folds, shuffle=True, random_state=seed).get_n_splits(X_train_new)

get_results([clfs[key] for key in list(clfs.keys()) if 'AVG' in key], 
            [key for key in list(clfs.keys()) if 'AVG' in key],
            X_train_new, y_train)
# Training models with updated dataframe

clfs['AVG Models 1'].fit(X_train_new, y_train)
clfs['AVG Models 2'].fit(X_train_new, y_train)
# Predicting the Sale Price

y_hat_model1 = np.expm1(clfs['AVG Models 1'].predict(X_test_new))
y_hat_model2 = np.expm1(clfs['AVG Models 2'].predict(X_test_new))
price_predict = (y_hat_model1*0.5) + (y_hat_model2*0.5)
solution = pd.DataFrame()
solution['Id'] = id_test
solution['SalePrice'] = price_predict
solution.to_csv('solution_final_v1.csv',index=False)
# View the distribution of SalePrice

fig, ax = plt.subplots(figsize=(8,5))
g = sns.distplot(price_predict, 
                 color="g",  
                 hist_kws=dict(edgecolor="w", linewidth=1), ax=ax);
plt.title("Predicted Sale Prices");
solution.head()
