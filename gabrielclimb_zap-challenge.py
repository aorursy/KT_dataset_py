print("teste")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")
df_train.shape
df_train.head()
df_train.dtypes
df_train.isnull().sum()
df_train.describe()
df_train.isnull().sum()/df_train.shape[0]
# As we can see, feature_0 has almostr 90% of null values, its not so usefull
df_train = df_train.drop("feature_0", axis=1)
# filling the null values with mean for those features with low std
df_train.feature_10.fillna(df_train.feature_10.mean(), inplace=True)
df_train.feature_12.fillna(df_train.feature_12.mean(), inplace=True)
df_train.feature_16.fillna(df_train.feature_16.mean(), inplace=True)
y = df_train.target
x = df_train.select_dtypes([float, int]).drop(["target", "id"], axis=1)
sns.pairplot(df_train.select_dtypes([float, int]).drop(["id"], axis=1))
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
corr = x.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
for c in df_train.select_dtypes(object).columns:
    print (c, len(df_train[c].value_counts()))
df_dummies = pd.DataFrame()

cols = ["feature_4"
        ,"feature_5"
        ,"feature_6"
        ,"feature_18"
        ,"feature_19"]

for c in cols:
    print (c)
    df_temp = pd.get_dummies(df_train[c], prefix=c)
    df_dummies = pd.concat([df_dummies, df_temp], axis=1)

x = pd.concat([x,df_dummies], axis=1)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=1)
xtrain.shape
%%time
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=10, random_state=14)
model = DecisionTreeRegressor(max_depth=11, random_state=14)
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, xtrain, ytrain, cv=kfold, scoring=scoring)
print(results.mean())

score = cross_val_score(model, xtest, ytest)
model.fit(xtrain,ytrain)
print ( model.score(xtest, ytest))


# feature_1
# -7.534562431913893
# 0.607581202092359
# CPU times: user 2min 45s, sys: 57.2 s, total: 3min 43s
# Wall time: 3min 43s

# feature_2
# -7.67435323758725
# 0.6122096565321538
# CPU times: user 3min 8s, sys: 1min 16s, total: 4min 25s
# Wall time: 4min 25s

# -7.7797591923655105
# 0.603857296344522
# CPU times: user 2.79 s, sys: 4 ms, total: 2.8 s
# Wall time: 2.79 s
%%time
import numpy as np # linear algebra

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt


estimators = np.arange(2, 200, 10)

model = RandomForestRegressor(n_jobs=-1)
scores = []
for n in estimators:
    model.set_params(n_estimators=n, warm_start=False)
    model.fit(xtrain, ytrain)
    scores.append(model.score(xtest, ytest))
#     print(model.score(xtest, ytest))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
from sklearn.model_selection import GridSearchCV
import xgboost

xgb = xgboost.XGBRegressor()
parameters = {'n_jobs':[-1],
              'objective':['reg:linear'],# 'reg:linear'
              'learning_rate': [0.01, 0.05, 0.07], #so called `eta` value
              'max_depth': [10, 12, 14],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.5],
              'n_estimators': [200]}

xgb_grid = GridSearchCV(xgb,
                        parameters,
                        cv = 40,
                        n_jobs = -1,
                        verbose=True)

xgb_grid.fit(xtrain,ytrain)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)





