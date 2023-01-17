import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
%matplotlib inline

random_seed = 42

from scipy import stats
from scipy.stats import skew

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, KFold
import xgboost as xgb
import lightgbm as lgb

import os
print(os.listdir("../input"))
# from https://stackoverflow.com/a/47167330
# To Impute Categorical Variables
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean',filler='NA'):
        self.strategy = strategy
        self.fill = filler

    def fit(self, X, y=None):
        if self.strategy in ['mean','median']:
            if not all(X.dtypes == np.number):
                raise ValueError('dtypes mismatch np.number dtype is \
                                 required for '+ self.strategy)
        if self.strategy == 'mean':
            self.fill = X.mean()
        elif self.strategy == 'median':
            self.fill = X.median()
        elif self.strategy == 'mode':
            self.fill = X.mode().iloc[0]
        elif self.strategy == 'fill':
            if type(self.fill) is list and type(X) is pd.DataFrame:
                self.fill = dict([(cname, v) for cname,v in zip(X.columns, self.fill)])
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

# from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# Base Models is a set of regressors which make predictions 
# The Meta Model combines the predictions to a final prediction
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_index], y.iloc[train_index])
                y_pred = instance.predict(X.iloc[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
    
def impute_and_to_dataframe(imputer, df, fit_end_index, cols):
    #Fit only on train set
    train = df[:fit_end_index][cols]
    imputer.fit(train)
    #Fill up full set
    imputed = imputer.transform(df[cols])
    return pd.DataFrame(imputed, columns=cols)

def plot_correlation(corr):
    sns.heatmap(corr, cmap="RdBu", square=True, vmin=-1.0, vmax=1.0)
    
def fill_null_with_mean(df):
    for c in df.columns:
        mean = df.loc[:,c].mean()
        x = df.loc[:,c].fillna(mean)
        df.loc[:,c] = x

def fill_null_with_mode(df):
    for c in df.columns:
        mode = df.loc[:,c].mode()[0]
        df.loc[:,c] = df.loc[:,c].fillna(mode)
        
def columns_into_encoded(df, columns, nofirst=True):
    result = df.copy()
    for c in columns:
        dummies = pd.get_dummies(full[c], prefix=c, drop_first=nofirst)
        result.drop(c, axis=1, inplace=True)
        result = pd.concat([result, dummies], axis = 1)
    return result

def calculate_skew(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    s = df[num_cols].apply(lambda x : skew(x.dropna())).sort_values(ascending=False)
    return pd.DataFrame({"Skew" : s})

def feature_ranking_regression(X, y):
    rfr = RandomForestRegressor()
    rfr.fit(X, y)
    importances = rfr.feature_importances_
    std = np.std([rfr.feature_importances_ for tree in rfr.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("{}. Feature {} ({}) Std Dev: {}".format(f + 1, X.columns[indices[f]], importances[indices[f]], std[indices[f]]))
        
def barplot_features(features, target, df, size=(20,20), ncols=2, xtickrot=90):
    cols = ncols
    rows = np.ceil(len(target)/cols)
    
    fig = plt.figure(figsize=size)
    for i, feature in enumerate(features):
        fig.add_subplot(rows, cols, i+1)
        plt.xticks(rotation=xtickrot)
        sns.barplot(x=feature, y=target, data=df)
        
def plot_features(df, features, target=None, plot="barplot", size=(20,20), ncols=2, xtickrot=90):
    cols = ncols
    rows = np.ceil(len(features)/cols)
    
    if plot != "distplot" and target == None:
        raise ValueError("Target Required")
    
    fig = plt.figure(figsize=size)
    for i, feature in enumerate(features):
        fig.add_subplot(rows, cols, i+1)
        plt.xticks(rotation=xtickrot)
        if plot == "barplot":
            sns.barplot(x=feature, y=target, data=df)
        elif plot == "distplot":
            sns.distplot(df[feature].dropna())
        else:
            raise ValueError("Unknown Plot type")
            
        
def print_scores(models, X_train, y_train, X_valid, y_valid):
    for model in models:
        print("Model: ", model, "Train Score ", model.score(X_train, y_train), "Test Score", model.score(X_valid, y_valid))
        
def plot_missing_value_ratios(df, size):
    xtickrot = 90
    missing_ratios = df.isnull().sum() / len(df) * 100
    missing_ratios = missing_ratios[missing_ratios != 0].sort_values(ascending=False)
    fig = plt.figure(figsize=size)
    plt.xticks(rotation=xtickrot)
    sns.barplot(x=missing_ratios.index, y=missing_ratios)
houses = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print("Datasets:", "houses:", houses.shape, "test:", houses.shape)
sns.lmplot(x="GrLivArea", y="SalePrice", data=houses, fit_reg=False)
houses = houses[houses.GrLivArea < 4000]
sns.lmplot(x="GrLivArea", y="SalePrice", data=houses)
sns.distplot(houses["SalePrice"])
stats.probplot(houses["SalePrice"], plot=plt)
plt.show()
forfree = houses[houses.SalePrice <= 0]
forfree.info()
log_prices = np.log(houses.SalePrice)
sns.distplot(log_prices)
stats.probplot(log_prices, plot=plt)
plt.show()
houses.SalePrice = log_prices
houses.describe()
full = pd.concat([houses, test], ignore_index=True)
full.shape
corr = houses.corr()

sale_price_corr = corr[["SalePrice"]]
sale_price_corr = sale_price_corr.drop("SalePrice")
sale_price_corr = sale_price_corr.sort_values(by="SalePrice", ascending = False)
sale_price_corr
plot_missing_value_ratios(full.drop(["SalePrice"], axis=1), size=(15,12))
full[["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]] = full[["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"]].fillna("NA")
full["LotFrontage"] = impute_and_to_dataframe(CustomImputer(strategy="median"), full, houses.shape[0], ["LotFrontage"])
full[["GarageType", "GarageFinish", "GarageQual", "GarageCond"]] = full[["GarageType", "GarageFinish", "GarageQual", "GarageCond"]].fillna("NA")
#TODO try mean or mode here
full["GarageYrBlt"] = impute_and_to_dataframe(CustomImputer(strategy="median"), full, houses.shape[0], ["GarageYrBlt"])
#TODO try "NA" here as filler
full["BsmtCond"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["BsmtCond"])
full["BsmtExposure"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["BsmtExposure"])
full["BsmtQual"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["BsmtQual"])
full["BsmtFinType1"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["BsmtFinType1"])
full["BsmtFinType2"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["BsmtFinType2"])
full["MasVnrType"] = full["MasVnrType"].fillna("None")
full["MasVnrArea"] = full["MasVnrArea"].fillna(0)
full["MSZoning"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["MSZoning"])

full = full.drop(["Utilities"], axis=1)

full["Functional"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["Functional"])
full["BsmtHalfBath"] = full["BsmtHalfBath"].fillna(0)
full["BsmtFullBath"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["BsmtFullBath"])
full["TotalBsmtSF"] = impute_and_to_dataframe(CustomImputer(strategy="mean"), full, houses.shape[0], ["TotalBsmtSF"])
full["SaleType"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["SaleType"])
full["KitchenQual"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["KitchenQual"])
full["GarageCars"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["GarageCars"])
full["GarageArea"] = impute_and_to_dataframe(CustomImputer(strategy="mean"), full, houses.shape[0], ["GarageArea"])
full["Exterior1st"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["Exterior1st"])
full["Exterior2nd"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["Exterior2nd"])
full["Electrical"] = impute_and_to_dataframe(CustomImputer(strategy="mode"), full, houses.shape[0], ["Electrical"])
full["BsmtUnfSF"] = impute_and_to_dataframe(CustomImputer(strategy="median"), full, houses.shape[0], ["BsmtUnfSF"])
full["BsmtFinSF1"] = full["BsmtFinSF1"].fillna(0)
full["BsmtFinSF2"] = full["BsmtFinSF2"].fillna(0)

full.info()
numeric_columns = houses.select_dtypes(include=[np.number]).columns
numeric_columns = numeric_columns.drop("SalePrice")
non_numeric_columns = houses.select_dtypes(include=["object"]).columns
#Total Square Feet
full["TotalSF"] = full["TotalBsmtSF"] + full["1stFlrSF"] + full["2ndFlrSF"]
skewed = calculate_skew(full)
skewed
safety_copy = full.copy()
#full = safety_copy
max_skew = 0.75

to_transform = [i for i, r in skewed.iterrows() if abs(r[0]) > max_skew]
for row in to_transform:
    full[row] = np.log1p(full[row])
full = pd.get_dummies(full, drop_first=True)
scaler = StandardScaler()
y_train_valid = houses.SalePrice
full.drop("SalePrice", axis=1, inplace=True)
X_train_valid = full[:houses.shape[0]]
X_test = full[houses.shape[0]:]
X_train_valid.loc[:,numeric_columns] = scaler.fit_transform(X_train_valid[numeric_columns])
X_test.loc[:,numeric_columns] = scaler.transform(X_test[numeric_columns])
X_train_valid.describe()
X_test.info()
def rmse_cv(model, folds=10):
    return np.sqrt(-cross_val_score(model, X_train_valid, y_train_valid, scoring="neg_mean_squared_error", cv=folds))

def print_rmse_cv_scores(models):
    for model in models:
        cv_results = rmse_cv(model)
        print("Model: ", model)
        print("Mean: ", format(cv_results.mean()))
        print("Median: ", format(np.median(cv_results)))
        print("Std Dev: ", format(cv_results.std()))
        print("=======================================")
lr = LinearRegression()
lr.fit(X_train_valid, y_train_valid)
ridge = Ridge()
ridge.fit(X_train_valid, y_train_valid)
lasso = Lasso()
lasso.fit(X_train_valid, y_train_valid)
en = ElasticNet()
en.fit(X_train_valid, y_train_valid)
svr = SVR(kernel="poly")
svr.fit(X_train_valid, y_train_valid)
kr = KernelRidge(kernel="polynomial")
kr.fit(X_train_valid,y_train_valid)
gboost = GradientBoostingRegressor(loss="huber")
gboost.fit(X_train_valid,y_train_valid)
xgboost = xgb.XGBRegressor()
xgboost.fit(X_train_valid,y_train_valid)
lgboost = lgb.LGBMRegressor()
lgboost.fit(X_train_valid, y_train_valid)
models = [lr, ridge, lasso, en, svr, kr, gboost, xgboost, lgboost]
#print_rmse_cv_scores(models)
stacked = StackingAveragedModels(base_models = [svr, kr, gboost, xgboost, lgboost], meta_model = ridge)
#rmse_cv(stacked)
stacked.fit(X_train_valid, y_train_valid)
stacked_pred = np.exp(stacked.predict(X_test))
xgboost.fit(X_train_valid,y_train_valid)
xgboost_pred = np.exp(xgboost.predict(X_test))
lgboost.fit(X_train_valid, y_train_valid)
lgboost_pred = np.exp(lgboost.predict(X_test))
y_test_pred = stacked_pred*1.00 + xgboost_pred*0.00 + lgboost_pred*0.00
sns.distplot(y_test_pred)
ids = full[houses.shape[0]:].Id
predictions = pd.DataFrame({"Id" : ids, "SalePrice" : y_test_pred})
predictions.to_csv("houses_predictions.csv", index=False)