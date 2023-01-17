# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from jcopml.plot import plot_correlation_matrix, plot_missing_value

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv", index_col="Id")
df.head()
df.shape
plot_missing_value(df, return_df=True).sort_values("missing_value", ascending=False).head(20)
df.drop(columns=["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "LotFrontage"], inplace=True)
df.shape
df['SalePrice'].isna().sum()
print(df["SalePrice"].describe())
plt.figure(figsize=(8, 4))
ax = sns.distplot(df["SalePrice"], bins=200)
filt = df["SalePrice"] < 400000
df = df.loc[filt]
df.shape
# New Histogram
print(df["SalePrice"].describe())
plt.figure(figsize=(8, 4))
ax = sns.distplot(df["SalePrice"], bins=200)
plot_correlation_matrix(df, "SalePrice")
correlation = df.corr()
n = 9
cols = correlation.nlargest(n, 'SalePrice')['SalePrice'].index
df_cor = np.corrcoef(df[cols].values.T)

plt.figure(figsize=(8, 5))
ax = sns.heatmap(df_cor, cbar=True, annot=True, fmt='.1f', yticklabels=cols.values, xticklabels=cols.values)
plt.show()
df_new = df[["YearBuilt", "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF", "FullBath", "SalePrice"]]
df_new.head(3)
sns.pairplot(df_new)
f, axes = plt.subplots(4, 2, figsize=(15, 15), sharex=False)
ax = sns.scatterplot(x="YearBuilt", y="SalePrice", data=df_new, hue="OverallQual", ax = axes[0, 0])
ax = sns.scatterplot(x="GrLivArea", y="SalePrice", data=df_new, hue="OverallQual", ax = axes[0, 1])
ax = sns.scatterplot(x="GarageArea", y="SalePrice", data=df_new, hue="OverallQual", ax = axes[1, 0])
ax = sns.scatterplot(x="TotalBsmtSF", y="SalePrice", data=df_new, hue="OverallQual", ax = axes[1, 1])
ax = sns.scatterplot(x="1stFlrSF", y="SalePrice", data=df_new, hue="OverallQual", ax = axes[2, 0])
ax = sns.boxplot(x="FullBath", y="SalePrice", data=df_new, ax=axes[2, 1])
ax = sns.boxplot(x="GarageCars", y="SalePrice", data=df_new, ax=axes[3, 0])
ax = sns.boxplot(x="OverallQual", y="SalePrice", data=df_new, ax=axes[3, 1])
f, axes = plt.subplots(4, 2, figsize=(15, 15), sharex=False)
ax = sns.distplot(df["YearBuilt"], bins=100, ax = axes[0, 0], color='r')
ax = sns.distplot(df["OverallQual"], bins=10, ax = axes[0, 1], color='g')
ax = sns.distplot(df["GrLivArea"], bins=100, ax = axes[1, 0], color='b')
ax = sns.distplot(df["GarageCars"], bins=5, ax = axes[1, 1])
ax = sns.distplot(df["GarageArea"], bins=100, ax = axes[2, 0])
ax = sns.distplot(df["TotalBsmtSF"], bins=100, ax = axes[2, 1], color='r')
ax = sns.distplot(df["1stFlrSF"], bins=100, ax = axes[3, 0], color='g')
ax = sns.distplot(df["FullBath"], bins=10, ax = axes[3, 1], color='b')
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease, mean_loss_decrease
# Separate the features and target columns
X = df_new.drop(columns=["SalePrice"])
y = df_new["SalePrice"]

# Create data train and data test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(poly=2), X_train.columns)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', RandomForestRegressor(n_jobs=-1, random_state=42))
])

parameter_tune = {
    'prep__numeric__poly__degree': [2, 3, 4],
    'prep__numeric__poly__interaction_only': [False, True],
    'algo__n_estimators': [220, 240],
    'algo__max_depth': [20, 50, 80],
    'algo__max_features': [0.5, 0.6, 0.8],
    'algo__min_samples_leaf': [1, 2, 3]
}

model_rf = RandomizedSearchCV(pipeline, parameter_tune, cv=3, n_iter=100,n_jobs=-1, verbose=1)
model_rf.fit(X_train, y_train)

print(model_rf.best_params_)
print(model_rf.score(X_train, y_train), model_rf.best_score_, model_rf.score(X_test, y_test))
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(poly=2), X_train.columns)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', RandomForestRegressor(n_jobs=-1, random_state=42))
])

parameter_tune = {
    'prep__numeric__poly__degree': [3],
    'prep__numeric__poly__interaction_only': [False, True],
    'algo__n_estimators': [260, 280],
    'algo__max_depth': [80, 100],
    'algo__max_features': [0.6, 0.8],
    'algo__min_samples_leaf': [2]
}

model_rf = GridSearchCV(pipeline, parameter_tune, cv=3, n_jobs=-1, verbose=1)
model_rf.fit(X_train, y_train)

print(model_rf.best_params_)
print(model_rf.score(X_train, y_train), model_rf.best_score_, model_rf.score(X_test, y_test))
df_imp = mean_score_decrease(X_train, y_train, model_rf, plot=True, topk=10)
from jcopml.plot import plot_actual_vs_prediction, plot_residual
plot_actual_vs_prediction(X_train, y_train, X_test, y_test, model_rf)
plot_residual(X_train, y_train, X_test, y_test, model_rf)
from xgboost import XGBRegressor
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(poly=2), X_train.columns)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', XGBRegressor(n_jobs=-1, random_state=42))
])

parameter_tune = {
    'algo__max_depth': [100],
    'algo__colsample_bytree': [0.8],
    'algo__n_estimators': [220],
    'algo__subsample': [0.8],
    'algo__gamma': [1, 5, 10],
    'algo__learning_rate': [0.01, 0.1, 1],
    'algo__reg_alpha': [0.01, 0.1],
    'algo__reg_lambda': [0.01, 0.1, 10] 
}

model_rf = GridSearchCV(pipeline, parameter_tune, cv=3, n_jobs=-1, verbose=1)
model_rf.fit(X_train, y_train)

print(model_rf.best_params_)
print(model_rf.score(X_train, y_train), model_rf.best_score_, model_rf.score(X_test, y_test))
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(poly=2), X_train.columns)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', XGBRegressor(n_jobs=-1, random_state=42))
])

parameter_tune = {
    'prep__numeric__poly__degree': [3],
    'prep__numeric__poly__interaction_only': [True, False],
    'algo__max_depth': [100],
    'algo__colsample_bytree': [0.8],
    'algo__n_estimators': [220],
    'algo__subsample': [0.8],
    'algo__gamma': [1, 2],
    'algo__learning_rate': [0.2],
    'algo__reg_alpha': [10],
    'algo__reg_lambda': [35] 
}

model_xgb = GridSearchCV(pipeline, parameter_tune, cv=3, n_jobs=-1, verbose=1)
model_xgb.fit(X_train, y_train)

print(model_xgb.best_params_)
print(model_xgb.score(X_train, y_train), model_xgb.best_score_, model_xgb.score(X_test, y_test))
df_imp = mean_score_decrease(X_train, y_train, model_xgb, plot=True, topk=10)
plot_actual_vs_prediction(X_train, y_train, X_test, y_test, model_xgb)
plot_residual(X_train, y_train, X_test, y_test, model_rf)