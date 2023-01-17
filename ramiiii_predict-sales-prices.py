import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
import os

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
train_df.head()
train_df.shape
train_df.isnull().sum()
train_df.drop(['Id', 'MiscFeature', 'Fence', 'PoolQC', 'Alley'], 1, inplace=True);
train_df.isnull().sum()
train_df.fillna(method='ffill', inplace=True)
train_df.info()
train_df.head()
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='45')

sns.barplot(x="MSSubClass", y="SalePrice", hue="SaleCondition", data=train_df)
fig, ax = plt.subplots(figsize=(16, 12))

ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])

plt.show()
t = train_df.iloc[:, [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 10, -2, -1]]

t.head()
sns.pairplot(t , hue='LotShape')
#Correlation map to see how features are correlated with SalePrice

corrmat = train_df.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

import xgboost as xgb
df = train_df.copy()

tf = test_df.copy()

for f in df.columns:

    if df[f].dtype == 'object':

        label = LabelEncoder()

        label.fit(list(df[f].values))

        df[f] = label.transform(list(df[f].values))

        

for f in tf.columns:

    if tf[f].dtype == 'object':

        label = LabelEncoder()

        label.fit(list(tf[f].values))

        tf[f] = label.transform(list(tf[f].values))



df.head()
X = df.drop(["SalePrice"],axis=1)

y = df['SalePrice']

X.shape
X_train, X_test, y_train ,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("model accuracy is:", model.score(X_test, y_test)*100)
test = tf.copy()

test.drop(['Id', 'MiscFeature', 'Fence', 'PoolQC', 'Alley'], 1, inplace=True);
test.head()
file_pred = model.predict(test)
sub=pd.DataFrame()



sub['Id'] = test_df['Id']

sub['SalePrice'] = file_pred



sub.to_csv('sample_submission.csv', index=False)

sub.head(10)