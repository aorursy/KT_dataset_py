# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

import seaborn as sns
iris = pd.read_csv("../input/Iris.csv", index_col=0)

iris.head()
sns.pairplot(iris, hue="Species");
knn1 = KNeighborsClassifier()

X_train = iris.drop("Species", axis=1).values

y_train = iris.Species.values

cross_val_score(knn1, X_train, y_train, cv=5)
select_params = {"n_neighbors": range(1, 11),

                 "weights": ["uniform", "distance"],

                 "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"]}

knn2 = KNeighborsClassifier()

grid_knn1 = GridSearchCV(knn2, select_params, cv=5)



grid_knn1.fit(X_train, y_train)
grid_knn1.best_params_
grid_knn1.best_score_
grid_results = grid_knn.cv_results_

for score, params in zip(grid_results["mean_test_score"], grid_results["params"]):

    print(score, params)
iris["PetalArea"] = iris["PetalWidthCm"] * iris["PetalLengthCm"]

iris["SepalArea"] = iris["SepalWidthCm"] * iris["SepalLengthCm"]
X_train2 = iris[["PetalArea", "SepalArea"]].values

y_train2 = iris.Species.values



knn3 = KNeighborsClassifier()

grid_knn2 = GridSearchCV(knn3, select_params, cv=5)

grid_knn2.fit(X_train2, y_train2)
grid_knn2.best_score_