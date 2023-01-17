# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from jcopml.plot import plot_missing_value

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease, mean_loss_decrease
from jcopml.tuning import grid_search_params as gsp
from jcopml.tuning import random_search_params as rsp
from jcopml.plot import plot_actual_vs_prediction, plot_residual

from xgboost import XGBRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/used-cars-price-prediction/train-data.csv")
df.drop(columns=["Unnamed: 0", "New_Price"], inplace=True)
df["Brand"] = df["Name"].str.split(" ").str[0]
df["Mileage"] = df["Mileage"].str.split(" ").str[0]
df["Engine"] = df["Engine"].str.split(" ").str[0]
df["Power"] = df["Power"].str.split(" ").str[0]
df.drop(columns="Name", inplace=True)
df = df.loc[df["Seats"]>0]
df = df[df['Mileage'].notna()]
df = df[df['Engine'].notna()]
df = df[df['Power'].notna()]
df = df[df['Seats'].notna()]
df = df.loc[df["Power"]!= 'null']
df["Mileage"] = pd.to_numeric(df["Mileage"])
df["Engine"] = pd.to_numeric(df["Engine"])
df["Power"] = pd.to_numeric(df["Power"])

df.head(4)
df.shape
plot_missing_value(df, return_df=True)
print(df["Price"].agg(["min", "max", "mean"]))
plt.figure(figsize=(12, 6))
ax = sns.distplot(df["Price"], bins=1000, color="red")
df.describe()
f, axes = plt.subplots(4, 2, figsize=(19, 15), sharex=False)
ax = sns.scatterplot(x="Year", y="Price", data=df, hue="Fuel_Type", size="Transmission", ax = axes[0, 0])
ax = sns.distplot(df["Year"], bins=19, ax = axes[0, 1], color='g')
ax = sns.scatterplot(x="Kilometers_Driven", y="Price", data=df, hue="Fuel_Type", size="Transmission", ax = axes[1, 0])
ax = sns.distplot(df["Kilometers_Driven"], bins=100, ax = axes[1, 1], color='r')
ax = sns.boxplot(x="Seats", y="Price", data=df, ax = axes[2, 0])
ax = sns.distplot(df["Seats"], bins=8, ax = axes[2, 1], color='r', kde_kws = {'bw' : 1.0})
ax = sns.scatterplot(x="Engine", y="Price", data=df, ax = axes[3, 0])
ax = sns.scatterplot(x="Power", y="Price", data=df, ax = axes[3, 1])
f, axes = plt.subplots(3, 2, figsize=(19, 15), sharex=False)
ax = sns.stripplot(x="Transmission", y="Price", data=df, ax=axes[0, 0])
ax = sns.stripplot(x="Location", y="Price", data=df, ax=axes[0, 1])
ax = sns.stripplot(x="Fuel_Type", y="Price", data=df, ax=axes[1, 0])
ax = sns.stripplot(x="Owner_Type", y="Price", data=df, ax=axes[1, 1])
ax = sns.stripplot(x="Brand", y="Price", data=df, ax=axes[2, 0])
correlation = df.corr()
n = 9
cols = correlation.nlargest(n, 'Price')['Price'].index
df_cor = np.corrcoef(df[cols].values.T)

plt.figure(figsize=(8, 5))
ax = sns.heatmap(df_cor, cbar=True, annot=True, fmt='.1f', yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# Separate the features and target columns
X = df.drop(columns=["Price"])
y = df["Price"]

# Create data train and data test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(poly=2, scaling='robust', transform='yeo-johnson'), ["Year", "Kilometers_Driven", "Mileage",
                                                                             "Engine", "Power", "Seats"]),
    ('categoric', cat_pipe(encoder='onehot'), ["Location", "Fuel_Type", "Transmission", "Owner_Type", "Brand"])
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', XGBRegressor(n_jobs=-1, random_state=42))
])

parameter_tune = {
    'algo__colsample_bytree': [0.9657264649759805], 
    'algo__gamma': [10], 
    'algo__learning_rate': [0.1], 
    'algo__max_depth': [10], 
    'algo__n_estimators': [200], 
    'algo__reg_alpha': [0.004359456845930351], 
    'algo__reg_lambda': [0.142428167785941], 
    'algo__subsample': [0.6], 
    'prep__numeric__poly__degree': [3], 
    'prep__numeric__poly__interaction_only': [True]
}

model_xgb = GridSearchCV(pipeline, parameter_tune, cv=3, n_jobs=-1, verbose=1)
model_xgb.fit(X_train, y_train)

print(model_xgb.best_params_)
print(model_xgb.score(X_train, y_train), model_xgb.best_score_, model_xgb.score(X_test, y_test))
print(" ")
print("Accuracy on train:", model_xgb.score(X_train, y_train))
print("Accuracy on test:", model_xgb.score(X_test, y_test))
mean_score_decrease(X_train, y_train, model_xgb, plot=True, topk=10)
plot_actual_vs_prediction(X_train, y_train, X_test, y_test, model_xgb)
plot_residual(X_train, y_train, X_test, y_test, model_xgb)
