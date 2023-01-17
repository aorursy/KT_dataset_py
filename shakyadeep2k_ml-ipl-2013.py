import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("../input/IPL2013.csv")

df.describe()
df.head()
df.columns
# Displaying first 10 attributes of top 5 instances

df.iloc[0:5, 0:10]
drop = ["Sl.NO.","PLAYER NAME", "TEAM"]

df.drop(drop, axis=1, inplace=True)

df.isnull().sum()
df.dtypes
features = list(df.columns)

cat_features = ['COUNTRY', 'TEAM', 'PLAYING ROLE']

num_features = []

for val in features:

    if val not in cat_features:

        num_features.append(val)

        

print(num_features)
# Plotting heatmap for only numerical features

df_copy = df[num_features]

corr_mat = df_copy.corr()

f, ax = plt.subplots(figsize=(12,10))

sns.heatmap(abs(corr_mat), ax=ax, cmap="Reds", linewidths = 0.1)
# One-hot Encoding 

df = pd.get_dummies(df)

df.columns
corr_mat = df.corr()

f, ax = plt.subplots(figsize=(12,10))

sns.heatmap(abs(corr_mat), ax=ax, cmap="Greens", linewidths = 0.1)
# Variance Inflation Factor

from statsmodels.stats.outliers_influence import variance_inflation_factor



def get_vif_factors(x):

    x_matrix = x.as_matrix()

    vif = [variance_inflation_factor(x_matrix,i) for i in range(x_matrix.shape[1])]

    vif_factors = pd.DataFrame()

    vif_factors['column'] = x.columns

    vif_factors['vif'] = vif

    return vif_factors



predictors = df.copy(deep=True)

predictors_copy = predictors[num_features]

predictors.drop(["SOLD PRICE"], axis=1, inplace=True)

predictors_copy.drop(["SOLD PRICE"], axis=1, inplace=True)

vif_factors = get_vif_factors(predictors_copy)

print(vif_factors)
large_vif = vif_factors[vif_factors.vif > 4].column

large_vif
predictors.drop(["ODI-SR-BL", "CAPTAINCY EXP"], axis=1, inplace=True)
predictors.columns
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(predictors, df["SOLD PRICE"], test_size=0.2)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train, y_train)

y_preds = reg.predict(X_test)
from sklearn.metrics import mean_absolute_error

MAE = mean_absolute_error(y_test, y_preds)
print("MAE score: ", MAE)