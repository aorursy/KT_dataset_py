# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
WINE_PATH = "../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv"
wine = pd.read_csv(WINE_PATH)
wine.shape
wine.info()
wine.describe()
%matplotlib inline

import matplotlib.pyplot as plt

wine.hist(bins=50, figsize=(20,15))

plt.show()
corr_matrix = wine.corr()
corr_matrix["quality"].sort_values(ascending=False)
val = wine.values

len(val[1])
from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(wine, test_size=0.2, random_state=42)
wine["alcohol_cat"] = pd.cut(wine["alcohol"],

                             bins=[0, 9.3, 10, 10.7, 11.4, 12.1, np.inf],

                             labels=[1, 2, 3, 4, 5, 6])

wine["alcohol_cat"].hist()
from sklearn.model_selection import StratifiedShuffleSplit



alcohol_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in alcohol_split.split(wine, wine["alcohol_cat"]):

    strat_train_set = wine.loc[train_index]

    strat_test_set = wine.loc[test_index]
wine = strat_train_set.copy()

from pandas.plotting import scatter_matrix



attributes = ["quality", "alcohol", "volatile acidity"]

scatter_matrix(wine[attributes], figsize=(15, 10))
wine = strat_train_set.drop(["quality"], axis=1)

wine_test = strat_test_set.drop(["quality"], axis=1)

wine_score_labels = strat_train_set["quality"].copy()

wine_test_score_labels = strat_test_set["quality"].copy()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer





num_pipeline = Pipeline([

    ("std_scaler", StandardScaler()),

    ])



wine_prepared = num_pipeline.fit_transform(wine)

wine_test_prepared = num_pipeline.transform(wine_test)
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor



lin_reg = LinearRegression()

lin_reg.fit(wine_prepared, wine_score_labels)



tree_reg = DecisionTreeRegressor(random_state=42)

tree_reg.fit(wine_prepared, wine_score_labels)
strat_test_set
from sklearn.metrics import mean_absolute_error
print("Linear Model: {}".format(lin_reg.score(wine_prepared, wine_score_labels)))

print("Decision Tree Model: {}".format(tree_reg.score(wine_prepared, wine_score_labels)))
from sklearn.metrics import mean_squared_error

wine_predictions = lin_reg.predict(wine_prepared)

lin_mse = mean_squared_error(wine_score_labels, wine_predictions)

line_rmse = np.sqrt(lin_mse)

line_rmse
def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())
from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_reg, wine_prepared, wine_score_labels,

                         scoring="neg_mean_squared_error", cv=30)

test_scores = cross_val_score(tree_reg, wine_test_prepared, wine_test_score_labels,

                         scoring="neg_mean_squared_error", cv=30)

tree_rmse_test_scores = np.sqrt(-test_scores)

tree_rmse_scores = np.sqrt(-scores)



scores = cross_val_score(lin_reg, wine_prepared, wine_score_labels,

                         scoring="neg_mean_squared_error", cv=30)

test_scores = cross_val_score(lin_reg, wine_test_prepared, wine_test_score_labels,

                         scoring="neg_mean_squared_error", cv=30)

lin_rmse_test_scores = np.sqrt(-test_scores)

lin_rmse_scores = np.sqrt(-scores)
print("Training:")

display_scores(tree_rmse_scores)

print("\nTesting:")

display_scores(tree_rmse_test_scores)
print("Training:")

display_scores(lin_rmse_scores)

print("\nTesting:")

display_scores(lin_rmse_test_scores)