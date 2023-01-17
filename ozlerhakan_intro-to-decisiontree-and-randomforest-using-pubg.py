# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pubg = pd.read_csv('../input/train_V2.csv')
pubg.head()
pubg.info()
# let's see the nOfNull values in each columns
pubg.isnull().sum()
pubg[pubg['winPlacePerc'].isnull()]
pubg.matchId.nunique()
pubg.matchType.value_counts()
pubg.describe()
def plot_heatmap(corrmat, title):
    sns.set(style = "white")
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Draw the heatmap with the mask and correct aspect ratio
    plt.figure(figsize=(20, 20))
    hm = sns.heatmap(corrmat, mask=mask, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, cmap=cmap)
    hm.set_title(title)
    plt.yticks(rotation=0)
    plt.show()
cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType']
cols_to_fit = [col for col in pubg.columns if col not in cols_to_drop]
corr = pubg[cols_to_fit].corr()
plot_heatmap(corr, "Correlation Table")
from pandas.plotting import scatter_matrix

scatter_matrix(pubg[["killPlace", "killPoints", "winPoints", "winPlacePerc"]], figsize=(16, 10));
pubg_copy = pubg.copy()
pubg_copy['winPlacePerc'].fillna(0, inplace=True)
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(pubg_copy, pubg_copy['matchType']):
    strat_train_set = pubg_copy.loc[train_index]
    strat_test_set = pubg_copy.loc[test_index]
X_train = strat_train_set[cols_to_fit[:-1]].copy()

y_train = strat_train_set.winPlacePerc.copy()
X_test = strat_test_set[cols_to_fit[:-1]].copy()

y_test = strat_test_set.winPlacePerc.copy()
X_train_dummy = pd.get_dummies(strat_train_set.matchType)
X_test_dummy = pd.get_dummies(strat_test_set.matchType)
X_train = pd.concat([X_train, X_train_dummy], axis='columns').copy()
X_test = pd.concat([X_test, X_test_dummy], axis='columns').copy()
# DecisionTree
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(X_train, y_train)

dt.score(X_train, y_train)
dt.score(X_test, y_test)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(DecisionTreeRegressor(), X_train, y_train, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

print("Scores:", tree_rmse_scores)
print("Mean:", tree_rmse_scores.mean())
print("Standard deviation:", tree_rmse_scores.std())
# RandomForest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=20)

rf.fit(X_train, y_train)

rf.score(X_train, y_train)
rf.score(X_test, y_test)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(RandomForestRegressor(n_estimators=20), X_train, y_train, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)

print("Scores:", forest_rmse_scores)
print("Mean:", forest_rmse_scores.mean())
print("Standard deviation:", forest_rmse_scores.std())
