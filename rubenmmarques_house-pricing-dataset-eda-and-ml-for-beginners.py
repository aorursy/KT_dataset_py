# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import math

import time



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.style as style



from scipy.stats import skew, boxcox_normmax

from scipy.special import boxcox1p



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

pd.options.display.max_columns = 500

pd.options.display.max_rows = 500
raw_train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

raw_test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



print(f"Train data: {raw_train_df.shape}")

print(f"Test data: {raw_test_df.shape}")
raw_train_df.sample(10)
raw_train_df.describe()
def custom_heatmap(df, include=[], exclude=[]):

    if include:

        df = df[include]

    elif exclude:

        df = df.drop(exclude, axis=1)

    fig, ax = plt.subplots(figsize=(30,20)) 

    correlation = df.corr()

    correlation[abs(correlation) < 0.1] = 0

    mask = np.zeros_like(correlation, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(correlation, ax=ax, cmap=sns.color_palette("RdBu_r", 100), mask=mask, annot=True)

    plt.show()

    

    

def custom_scatterplot(df, features=[], target=None, numcols=3):

    style.use('seaborn')

    for i, feature in enumerate(features):

        fig = plt.figure(i//numcols,figsize=(25, 5))

        ax = plt.subplot(1, numcols, (i%numcols)+1)

        ax.scatter(df[feature], target, alpha=0.2)

        ax.set_title(feature)

        ax.set_ylabel(target.name)  

    plt.tight_layout()

    plt.show()

    

    

def custom_distributionplot(df, columns, numcols=3):

    style.use('seaborn')

    for i, col in enumerate(columns):

        fig = plt.figure(i//numcols,figsize=(25, 5))

        ax = plt.subplot(1, numcols, (i%numcols)+1)

        sns.distplot(df[col], ax=ax)

        ax.set_title(col)

    plt.tight_layout()

    plt.show()  

    

    

def custom_boxplot(df, features=[], target=None, numcols=4):

    style.use('seaborn')

    for i, feature in enumerate(features):

        fig = plt.figure(i//numcols,figsize=(25, 5))

        ax = plt.subplot(1, numcols, (i%numcols)+1)

        data = pd.concat([df[feature], target], axis=1)

        sns.boxplot(x=df[feature], y=target, data=data, ax=ax)

        ax.set_title(feature)

        plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

    

def custom_countplot(df, features=[], numcols=4):

    style.use('seaborn')

    for i, feature in enumerate(features):

        fig = plt.figure(i//numcols,figsize=(25, 5))

        ax = plt.subplot(1, numcols, (i%numcols)+1)

        sns.countplot(df[feature], ax=ax)

        ax.set_title(feature)

        plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

    

    

def df_nan_percentage(df):

    total = df.isnull().sum()[ df.isnull().sum() > 0 ].sort_values(ascending=False)

    percent = (total/df.shape[0]) * 100

    new_df = pd.DataFrame({

        "Total": total,

        "Percentage": percent

    }, index=total.index)

    return new_df





def BooleanEncoding(df, features=[], threshold=0):

    names = []

    new_features = []

    

    for feature in features:

        names.append(f"has{feature}")

        new_features.append(df[feature].map(lambda x: 1 if x > threshold else 0))

        

    new_df = pd.concat(new_features, axis=1)

    new_df.columns = names

    return new_df





def check_outliers(df, method="z-score"):

    if method == "z-score":

        z = np.abs(stats.zscore(np.log1p(df)))

        outliers = np.where(z>3)[0]

        

    elif method == "iqr-score":

        Q1 = df.quantile(0.25)

        Q3 = df.quantile(0.75)

        IQR = Q3 - Q1

        outliers = df[((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)].index

    

    return outliers





def check_overfitting(df, threshold=0.05):

    overfit = []

    for i in df.columns:

        counts = df[i].value_counts()

        zeros = counts.iloc[0]

        if zeros / len(df) > (1 - threshold):

            overfit.append(i)

    return overfit



    

def correct_skewness(df, features):

    for feat in features:

        try:

            df.loc[:, feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))

        except Exception as e:

            print(e)

    return df





def adjusted_r2(r2,n,k):

    return r2-(k-1)/(n-k)*(1-r2)

raw_train_df
target_features = ["Id", "SalePrice"]



area_features = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",

                 "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea"]



nominal_features = ["MSSubClass",  "MSZoning", "Street", "Alley", "LotShape", "LandContour", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",

                    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", "Electrical", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",

                    "Functional", "GarageType", "GarageFinish", "PavedDrive", "MiscFeature", "MoSold", "YrSold", "SaleType", "SaleCondition"]



ordinal_features = ["Utilities", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",

                    "HeatingQC", "CentralAir", "BedroomAbvGr", "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd", "Fireplaces", "FireplaceQu", "GarageYrBlt", "GarageCars", "GarageQual", "GarageCond", 

                    "PoolQC", "Fence", "MiscVal"]



print(f"Total features: {len(area_features)+ len(nominal_features)+len(ordinal_features)}")
custom_scatterplot(raw_train_df, features=area_features, target=raw_train_df["SalePrice"], numcols=4)
raw_train_df[area_features].describe()
custom_boxplot(raw_train_df[nominal_features], features=nominal_features, target=raw_train_df["SalePrice"])
custom_countplot(raw_train_df[nominal_features], features=nominal_features)
custom_boxplot(raw_train_df[ordinal_features], features=ordinal_features, target=raw_train_df["SalePrice"])
custom_countplot(raw_train_df[ordinal_features], features=ordinal_features)
train_df = raw_train_df.copy()
custom_distributionplot(train_df, columns=["SalePrice"])
train_df["SalePrice"] = np.log1p(raw_train_df["SalePrice"])
custom_distributionplot(train_df, columns=["SalePrice"])
outliers = [30, 88, 462, 631, 1322]
train_df = train_df[

    (train_df["SalePrice"] >= train_df["SalePrice"].quantile(0.01))&

    (train_df["SalePrice"] <= train_df["SalePrice"].quantile(0.99))&

    ~(train_df["Id"].isin(outliers))

]
custom_distributionplot(train_df, columns=["SalePrice"])
df_nan_percentage(train_df)
train_df[["GarageYrBlt", "GarageType", "GarageFinish", "GarageQual", "GarageCond"]] = train_df[["GarageYrBlt", "GarageType", "GarageFinish", "GarageQual", "GarageCond"]].fillna("None")

train_df[["BsmtFinType2", "BsmtExposure", "BsmtFinType1", "BsmtCond", "BsmtQual"]] = train_df[["BsmtFinType2", "BsmtExposure", "BsmtFinType1", "BsmtCond", "BsmtQual"]].fillna("None")

train_df[["PoolQC", "Fence", "FireplaceQu", "Alley", "MiscFeature"]] = train_df[["PoolQC", "Fence", "FireplaceQu", "Alley", "MiscFeature"]].fillna("None")



train_df["LotFrontage"] = train_df["LotFrontage"].fillna(train_df["LotFrontage"].median())

train_df["MasVnrArea"] = train_df["MasVnrArea"].fillna(train_df["MasVnrArea"].median())



train_df["MasVnrType"] = train_df["MasVnrType"].fillna(train_df["MasVnrType"].mode()[0])

train_df["Electrical"] = train_df["Electrical"].fillna(train_df["Electrical"].mode()[0])
restore_train_df = train_df.copy()

df_nan_percentage(train_df)
# create checkpoint

train_df = restore_train_df.copy()

target_df = train_df[["Id", "SalePrice"]]
# These features should be categorical and not numerical

train_df['MSSubClass'] = train_df['MSSubClass'].apply(str)

train_df['YrSold'] = train_df['YrSold'].astype(str)

train_df['MoSold'] = train_df['MoSold'].astype(str)
def create_new_features(df):

    mapping = {

        "Ex": 10,

        "Gd": 8,

        "TA": 6,

        "Fa": 4,

        "Po": 2,

        "None": 0

    }



    TotalPorchArea = df[["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]].sum(axis=1)

    TotalPorchArea.name = "TotalPorchArea"



    TerrainArea = df["LotArea"] - ( TotalPorchArea + df[["LotFrontage", "GrLivArea", "TotalBsmtSF", "GarageArea", "PoolArea"]].sum(axis=1))

    TerrainArea.name = "TerrainArea"

    

    TotalBathrooms = df[["FullBath", "BsmtFullBath"]].sum(axis=1) + 0.5*df[["HalfBath", "BsmtHalfBath"]].sum(axis=1)

    TotalBathrooms.name = "TotalBathrooms"

    

    CustomQual = df[["ExterQual", "BsmtQual", "GarageQual", "PoolQC", "KitchenQual", "FireplaceQu", "HeatingQC"]].applymap(lambda x: mapping[x] if x in mapping else x)

    CustomQual = CustomQual.sum(axis=1) / CustomQual[CustomQual > 0].count(axis=1)

    CustomQual = pd.concat([df["OverallQual"], CustomQual], axis=1)

    CustomQual = CustomQual.mean(axis=1)

    CustomQual.name = "CustomQual"

    

    

    CustomCond = df[["ExterCond", "BsmtCond", "GarageCond"]].applymap(lambda x: mapping[x] if x in mapping else x)

    CustomCond = CustomCond.sum(axis=1) / CustomCond[CustomCond > 0].count(axis=1)

    CustomCond = pd.concat([df["OverallCond"], CustomCond], axis=1)

    CustomCond = CustomCond.mean(axis=1)

    CustomCond.name = "CustomCond"



    new_features = pd.concat([TotalPorchArea, TerrainArea, TotalBathrooms, CustomQual, CustomCond], axis=1)

    return new_features



new_features = create_new_features(train_df.copy())

new_features
def create_new_boolean_features(df):

    mapping = {

        "Ex": 10,

        "Gd": 8,

        "TA": 6,

        "Fa": 4,

        "Po": 2,

        "None": 0

    }



    MaterialsUsed = pd.get_dummies(df[["Foundation", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType"]])

    #MaterialsUsed = MaterialsUsed.T.groupby([s.split('_')[1] for s in MaterialsUsed.T.index.values]).sum().T

    

    Style = pd.get_dummies(df[["HouseStyle", "RoofStyle", "BldgType", "LotShape", "LotConfig", "GarageType"]])

    #Style = Style.T.groupby([s.split('_')[1] for s in Style.T.index.values]).sum().T



    Surroundings = pd.get_dummies(df[["Alley", "Street", "LandContour", "LandSlope", "Condition1", "Condition2", "MSZoning", "Neighborhood"]])

    #Surroundings = Surroundings.T.groupby([s.split('_')[1] for s in Surroundings.T.index.values]).sum().T



    Utilities = pd.get_dummies(df[["Electrical", "Heating"]])

    #Utilities = Utilities.T.groupby([s.split('_')[1] for s in Utilities.T.index.values]).sum().T



    others = pd.get_dummies(df[["MSSubClass", "YrSold", "MoSold"]])

    

    new_boolean_features = BooleanEncoding(df, features=["MasVnrArea", "LowQualFinSF", "BsmtFinSF2", "2ndFlrSF", "WoodDeckSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "Fireplaces"], threshold=0)

    

    hasCentralAir = df["CentralAir"].map({"N":0, "Y": 1})

    hasCentralAir.name = "hasCentralAir"

    

    hasExFireplace = df["FireplaceQu"].map(lambda x: 1 if x == "Ex" else 0)

    hasExFireplace.name = "hasExFireplace"

    hasExPool = df["PoolQC"].map(lambda x: 1 if x == "Ex" else 0)

    hasExPool.name = "HasExPool"

    hasExKitchenQual = df["KitchenQual"].map(lambda x: 1 if x == "Ex" else 0)

    hasExKitchenQual.name = "hasExKitchenQualQC"

    hasExHeatingQC = df["HeatingQC"].map(lambda x: 1 if x == "Ex" else 0)

    hasExHeatingQC.name = "hasHeatingQC"

    hasBsmtExposure = df["BsmtExposure"].map(lambda x: 0 if x == "No" else 1)

    hasBsmtExposure.name = "hasBsmtExposure"

    

    new_features = pd.concat([MaterialsUsed, Style, Utilities, others, new_boolean_features, hasCentralAir, hasExFireplace, hasExPool, hasExKitchenQual, hasExHeatingQC, hasBsmtExposure], axis=1)

    return new_features



new_boolean_features = create_new_boolean_features(train_df.copy())

new_boolean_features
boolean_features_dist = (new_boolean_features.sum() / new_boolean_features.shape[0])

boolean_features_dist
# Chossing boolean features that are balanced

useful_boolean_features = boolean_features_dist[

    (boolean_features_dist > 0.3) & 

    (boolean_features_dist < 0.7)

].index.values
train_df[new_features.columns] = new_features

train_df
train_df.describe()
train_df.skew()
skew_features = train_df.skew()[abs(train_df.skew()) > 0.5]

skew_features
new_skew_df = correct_skewness(train_df.copy(), skew_features.index)

new_skew_df
drop_skewed_features = new_skew_df.skew()[abs(new_skew_df.skew()) > 0.5]

train_df[skew_features.index] = new_skew_df[skew_features.index]

train_df = train_df.drop(drop_skewed_features.index, axis=1)

train_df.skew()
train_df[useful_boolean_features] = new_boolean_features[useful_boolean_features]

train_df
# Remove features that have more than 90% zeros

overfitting_features = check_overfitting(train_df, threshold=0.1)

overfitting_features
# Drop Columns not needed

train_df.drop(overfitting_features, axis=1, inplace=True)

train_df.drop(train_df.select_dtypes("object").columns, axis=1, inplace=True)

train_df.drop(target_df.columns, axis=1, inplace=True)

train_df.drop(["YearRemodAdd"], axis=1, inplace=True)
df_nan_percentage(train_df)
train_df
train_df.columns
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler



class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns, copy=True, with_mean=True, with_std=True):

        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)

        self.columns = columns

        self.mean_ = None

        self.var_ = None

        

    def fit(self, X, y=None):

        self.scaler.fit(X[self.columns], y)

        self.mean_ = np.mean(X[self.columns])

        self.var_ = np.var(X[self.columns])

        return self

    

    def transform(self, X, y=None, copy=None):

        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)

        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]

        return pd.concat([X_not_scaled, X_scaled], axis=1)
unscaled_train_df = train_df.copy()



scale_features = [

    'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',

    'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 

    'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars','GarageArea', 

    'TotalPorchArea', 'TotalBathrooms', 'CustomQual', 'CustomCond'

]



scaler = CustomScaler(scale_features)

scaler.fit(unscaled_train_df)
target_df.reset_index(drop=True, inplace=True)

scaled_features = scaler.transform(unscaled_train_df.reset_index(drop=True))

scaled_features
scaled_train_df = scaled_features.copy()

custom_scatterplot(scaled_train_df, features=train_df.columns, target=target_df["SalePrice"])
custom_heatmap(pd.concat([scaled_train_df, pd.DataFrame(target_df["SalePrice"])], axis=1))
correlation_df = pd.concat([scaled_train_df, target_df["SalePrice"]], axis=1).corr()
final_features = correlation_df.loc[:, "SalePrice"][

    (correlation_df.loc[:, "SalePrice"] > 0.1) |

    (correlation_df.loc[:, "SalePrice"] < -0.1)

].index.values



final_features = final_features[np.where(final_features != "SalePrice")]

final_features
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, StratifiedKFold, train_test_split, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, LarsCV, LassoLarsCV

from sklearn.svm import SVR

import xgboost as xgb

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline
# model scoring and validation function

def cv_rmse(model, X, y):

    rmse = np.sqrt(-cross_val_score(model, X, y,scoring="neg_mean_squared_error",cv=kfolds))

    return (rmse)



# rmsle scoring function

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def inverse_target(target, scale):

    #expm1 is the inverse of logp1

    return pd.DataFrame(np.expm1(target/scale), columns=["SalePrice"]).apply(np.ceil)



def plot_predictions(y_train, models, predictions, scores):

    nrows = len(models)

    sns.set_style("white")

    fig, axs = plt.subplots(ncols=0, nrows=nrows, figsize=(8, 7))

    plt.subplots_adjust(top=3.5, right=2)

    for i, model in enumerate(models, 1):

        plt.subplot(nrows, 1, i)

        plt.scatter(predictions[model], np.expm1(y_train))

        plt.plot([0, 800000], [0, 800000], '--r')



        plt.xlabel('{} Predictions (y_pred)'.format(model), size=15)

        plt.ylabel('Real Values (y_train)', size=13)

        plt.tick_params(axis='x', labelsize=12)

        plt.tick_params(axis='y', labelsize=12)



        plt.title('{} Predictions vs Real Values'.format(model), size=15)

        plt.text(0, 700000, 'Mean RMSE: {:.6f} / Std: {:.6f}'.format(scores[model][0], scores[model][1]), fontsize=15)

        #ax.xaxis.grid(False)

        sns.despine(trim=True, left=True)

    plt.show()



    

def predict(models, X, y):

    predictions = {}

    scores = {}

    for name, model in models.items():

    

        model.fit(X, y.values.ravel())

        predictions[name] = np.expm1(model.predict(X))

    

        score = cv_rmse(model, X=X, y=y.values.ravel())

        scores[name] = score

        

    return predictions,scores

ids = target_df["Id"]

X_train = scaled_train_df.copy()[final_features]

y_train = target_df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.8)

print(f"X train: {X_train.shape}")

print(f"y train: {y_train.shape}")

print(f"X test: {X_test.shape}")

print(f"y test: {y_test.shape}")
kfolds = KFold(n_splits=8, shuffle=True)



## Ridge Model

ridge_alphas = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

ridge = RidgeCV(alphas=ridge_alphas, cv=kfolds)



## Lasso Model

lasso_alphas = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

lasso = LassoCV(max_iter=1e7, alphas=lasso_alphas, cv=kfolds)



## Lars Model

lars = LarsCV(max_iter=500, max_n_alphas=500, cv=kfolds)



## ElasticNet Model

elasticnet_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

elasticnet_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

elasticnet = ElasticNetCV(max_iter=1e7, alphas=elasticnet_alphas, cv=kfolds, l1_ratio=elasticnet_l1ratio)





linear_models = {

    'Ridge': ridge,

    'Lasso': lasso, 

    'ElasticNet': elasticnet,

    'Lars': lars

}
predictions,scores = predict(linear_models, X_train, y_train)

for name,score in scores.items():

    print("{} score: {:.4f} ({:.4f})\n".format(name, score.mean(), score.std()))
plot_predictions(y_train, linear_models, predictions, scores)
predictions, scores = predict(linear_models, X_test, y_test)

for name,score in scores.items():

    print("{} score: {:.4f} ({:.4f})\n".format(name, score.mean(), score.std()))
params = {

    'random_state': [1, 2, 3, 4, 5],

    'bootstrap': [True, False],

    'max_features': ["auto", "sqrt", "log2"], 

    'n_estimators': [120 , 140, 160, 180, 200, 220, 240, 260, 280, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]

}
rf_model = RandomForestRegressor()

gs_rf = RandomizedSearchCV(rf_model, params, cv=kfolds, scoring='neg_mean_squared_error')

gs_rf.fit(X_train, y_train)

print("Scores:\n", gs_rf.best_score_)

print("Best Params:\n", gs_rf.best_params_)
xgb_model = xgb.XGBRegressor(objective="reg:squarederror")

gs_xgb = RandomizedSearchCV(xgb_model, params, cv=kfolds, scoring='neg_mean_squared_error')

gs_xgb.fit(X_train, y_train)

print("Scores:\n", gs_xgb.best_score_)

print("Best Params:\n", gs_xgb.best_params_)
rf_model = RandomForestRegressor(**gs_rf.best_params_)

xgb_model = xgb.XGBRegressor(objective="reg:squarederror", **gs_xgb.best_params_)
ensemble_models = {

    "rf": rf_model,

    "xgb": xgb_model

}
predictions, scores = predict(ensemble_models, X_test, y_test)

for name,score in scores.items():

    print("{} score: {:.4f} ({:.4f})\n".format(name, score.mean(), score.std()))
X = raw_test_df.copy()

X
class HousePricingModel():

    

    def __init__(self):

        self.used_features = final_features

        self.scaler = CustomScaler([

            "LotFrontage", "LotArea", "OverallQual", "BsmtFinSF1", 

            'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 

            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 

            'GarageArea', 'TotalPorchArea', 'TotalBathrooms',

            'CustomQual', 'YearBuilt'

        ])

        

        self.data = None

    

    def __area_features_engineering(self, df):

        mapping = {

            "Ex": 10,

            "Gd": 8,

            "TA": 6,

            "Fa": 4,

            "Po": 2,

            "None": 0

        }

        

        TotalPorchArea = df[["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]].sum(axis=1)

        TotalPorchArea.name = "TotalPorchArea"



        TerrainArea = df["LotArea"] - ( TotalPorchArea + df[["LotFrontage", "GrLivArea", "TotalBsmtSF", "GarageArea", "PoolArea"]].sum(axis=1))

        TerrainArea.name = "TerrainArea"



        TotalBathrooms = df[["FullBath", "BsmtFullBath"]].sum(axis=1) + 0.5*df[["HalfBath", "BsmtHalfBath"]].sum(axis=1)

        TotalBathrooms.name = "TotalBathrooms"



        CustomQual = df[["ExterQual", "BsmtQual", "GarageQual", "PoolQC", "KitchenQual", "FireplaceQu", "HeatingQC"]].applymap(lambda x: mapping[x] if x in mapping else x)

        CustomQual = CustomQual.sum(axis=1) / CustomQual[CustomQual > 0].count(axis=1)

        CustomQual = pd.concat([df["OverallQual"], CustomQual], axis=1)

        CustomQual = CustomQual.mean(axis=1)

        CustomQual.name = "CustomQual"





        CustomCond = df[["ExterCond", "BsmtCond", "GarageCond"]].applymap(lambda x: mapping[x] if x in mapping else x)

        CustomCond = CustomCond.sum(axis=1) / CustomCond[CustomCond > 0].count(axis=1)

        CustomCond = pd.concat([df["OverallCond"], CustomCond], axis=1)

        CustomCond = CustomCond.mean(axis=1)

        CustomCond.name = "CustomCond"



        new_features = pd.concat([TotalPorchArea, TerrainArea, TotalBathrooms, CustomQual, CustomCond], axis=1)

        return new_features

    

    def __boolean_features_engineering(self, df):

        mapping = {

            "Ex": 10,

            "Gd": 8,

            "TA": 6,

            "Fa": 4,

            "Po": 2,

            "None": 0

        }



        MaterialsUsed = pd.get_dummies(df[["Foundation", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType"]])

        Style = pd.get_dummies(df[["HouseStyle", "RoofStyle", "BldgType", "LotShape", "LotConfig", "GarageType"]])

        Surroundings = pd.get_dummies(df[["Alley", "Street", "LandContour", "LandSlope", "Condition1", "Condition2", "MSZoning", "Neighborhood"]])

        Utilities = pd.get_dummies(df[["Electrical", "Heating"]])

        others = pd.get_dummies(df[["MSSubClass", "YrSold", "MoSold"]])



        new_boolean_features = BooleanEncoding(

            df, 

            features=["MasVnrArea", "LowQualFinSF", "BsmtFinSF2", "2ndFlrSF", "WoodDeckSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "Fireplaces"], 

            threshold=0

        )

        

        # Boolean Encoding for categorical variables

        hasCentralAir = df["CentralAir"].map({"N":0, "Y": 1})

        hasCentralAir.name = "hasCentralAir"

        hasExFireplace = df["FireplaceQu"].map(lambda x: 1 if x == "Ex" else 0)

        hasExFireplace.name = "hasExFireplace"

        hasExPool = df["PoolQC"].map(lambda x: 1 if x == "Ex" else 0)

        hasExPool.name = "HasExPool"

        hasExKitchenQual = df["KitchenQual"].map(lambda x: 1 if x == "Ex" else 0)

        hasExKitchenQual.name = "hasExKitchenQualQC"

        hasExHeatingQC = df["HeatingQC"].map(lambda x: 1 if x == "Ex" else 0)

        hasExHeatingQC.name = "hasHeatingQC"

        hasBsmtExposure = df["BsmtExposure"].map(lambda x: 0 if x == "No" else 1)

        hasBsmtExposure.name = "hasBsmtExposure"



        new_features = pd.concat([MaterialsUsed, Style, Utilities, others, new_boolean_features, hasCentralAir, hasExFireplace, hasExPool, hasExKitchenQual, hasExHeatingQC, hasBsmtExposure], axis=1)

        return new_features

    

    

    def __correct_skewness(self, df):

        new_df = df.copy()

        features = new_df.columns

        for feat in features:

            try:

                new_df.loc[:, feat] = boxcox1p(new_df[feat], boxcox_normmax(new_df[feat] + 1))

            except Exception as e:

                print(f"{feat} {e}")

        

        return new_df

        



    def predict(self, model):

        if self.data is not None:

            pred_outputs = model.predict(self.data)

            return np.expm1(pred_outputs)

        

        

    def predicted_outputs(self, model):

        if self.data is not None:

            self.preprocessed_data["Probability"] = model.predict_proba(self.data)[:, 1]

            self.preprocessed_data["Prediction"] = model.predict(self.data)

            return self.preprocessed_data

        

        

    def load_and_clean_data(self, df):

        # import the data

        #df = pd.read_csv(data_file, delimiter=",")



        fill_features_with_none = [

            "GarageYrBlt", "GarageType", "GarageFinish", "GarageQual", "GarageCond", 

            "BsmtFinType2", "BsmtExposure", "BsmtFinType1", "BsmtCond", "BsmtQual", 

            "PoolQC", "Fence", "FireplaceQu", "Alley", "MiscFeature", "KitchenQual"

        ]

        df[fill_features_with_none] = df[fill_features_with_none].fillna("None")



        fill_features_with_mode = [

            "MasVnrType", "Electrical", "SaleType", "MSZoning", "Utilities", 

            "Exterior1st", "Exterior2nd", "GarageCars", "Functional"

        ]

        for feature in fill_features_with_mode:

            df[feature] = df[feature].fillna(df[feature].mode()[0])

        

        fill_features_with_median = [

            "BsmtHalfBath", "BsmtFullBath", "TotalBsmtSF", "BsmtUnfSF", "BsmtFinSF2", 

            "BsmtFinSF1", "LotFrontage", "MasVnrArea", "GarageArea"

        ]

        for feature in fill_features_with_median:

            df[feature] = df[feature].fillna(df[feature].mode()[0])

        

        # convert numerical data to categorical data

        df['MSSubClass'] = df['MSSubClass'].apply(str)

        df['YrSold'] = df['YrSold'].astype(str)

        df['MoSold'] = df['MoSold'].astype(str)

        

        # feature engineering

        new_area_features = self.__area_features_engineering(df.copy()) 

        new_boolean_features = self.__boolean_features_engineering(df.copy())        

        final_features = list(set(list(df.columns.values) + list(new_area_features.columns.values) + list(new_boolean_features.columns.values)))

        final_df = pd.concat([df, new_area_features, new_boolean_features], axis=1)

        self.ids = final_df["Id"]

        final_df.drop(["Id"], axis=1, inplace=True)

        final_df.drop(final_df.select_dtypes("object").columns, axis=1, inplace=True)

        final_df = final_df[self.used_features]

        

        #skweness

        skewless_df = self.__correct_skewness(final_df.drop(new_boolean_features.columns, axis=1, errors='ignore')) 

        final_df[skewless_df.columns] = skewless_df 

      

        #scaling

        self.scaler.fit(final_df)

        scaled_data = self.scaler.transform(final_df.copy().reset_index(drop=True))



        self.preprocessed_data = scaled_data.copy()

        self.data = self.preprocessed_data.copy()[self.used_features]
models = {**linear_models, **ensemble_models}
data_model = HousePricingModel()

data_model.load_and_clean_data(X) #Data must be positive



predicted_values = {}

for key,value in models.items():

    y_hat = data_model.predict(value)

    predicted_values[key] = y_hat
predicted_values
#avering values of each model

mean_predicted_values = np.mean(list(predicted_values.values()), axis=0)

mean_predicted_values
submission = pd.DataFrame()

submission = submission.assign(id=X["Id"], SalePrice=mean_predicted_values)

submission
submission.to_csv(f"/kaggle/working/submission_{str(round(time.time()))}.csv", index=False)