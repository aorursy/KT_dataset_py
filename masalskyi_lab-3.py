# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv")
df.describe()
df.info()
df.head()
sns.catplot(x="churn", kind="count", palette="ch:.25", data=df)
df["area code"].value_counts()
df["state"].value_counts()
df_1 = df.drop(["state","phone number"], axis = 1)
y = df_1["churn"].map({False:0,True:1})
df_2 = df_1.drop(["area code", "churn"], axis = 1)
from sklearn.metrics import f1_score, confusion_matrix

df_2["international plan"] =df_2["international plan"].map({"no":0, "yes":1})
df_2["voice mail plan"] =df_2["voice mail plan"].map({"no":0, "yes":1})
X = df_2

X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print("F1 score: ",f1_score(y_pred, y_test))
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
import numpy as np

params = {"max_depth":np.arange(2,21)}

from sklearn.model_selection import GridSearchCV

tree_grid_depth = GridSearchCV(tree, params, cv=kf, scoring='f1')

tree_grid_depth.fit(X, y)

tree_grid_depth.best_params_
params = {"min_samples_split":np.arange(2,21)}

tree = DecisionTreeClassifier(max_depth = 6)

tree_grid_samples_split = GridSearchCV(tree, params, cv=kf, scoring='f1')

tree_grid_samples_split.fit(X, y)

tree_grid_samples_split.best_params_
params = {"min_samples_leaf":np.arange(2,21)}

tree = DecisionTreeClassifier(max_depth = 6, min_samples_split = 11)

tree_grid_samples_leaf = GridSearchCV(tree, params, cv=kf, scoring='f1')

tree_grid_samples_leaf.fit(X, y)

tree_grid_samples_leaf.best_params_
X.shape
params = {"max_features":np.arange(2,18)}

tree = DecisionTreeClassifier(max_depth = 6, min_samples_split = 11, min_samples_leaf = 3)

tree_grid_features = GridSearchCV(tree, params, cv=kf, scoring='f1')

tree_grid_features.fit(X, y)

tree_grid_features.best_params_
results_depth = pd.DataFrame(tree_grid_depth.cv_results_)

result_samples_split = pd.DataFrame(tree_grid_samples_split.cv_results_)

result_samples_leaf = pd.DataFrame(tree_grid_samples_leaf.cv_results_)

result_features = pd.DataFrame(tree_grid_features.cv_results_)
import matplotlib.pyplot as plt

fig, ((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2, figsize = (10,10))



ax11.set_xlabel("max_depth")

ax11.set_ylabel("f1 score")

ax11.plot(results_depth["param_max_depth"], results_depth["mean_test_score"])



ax12.set_xlabel("min_samples_split")

ax12.plot(result_samples_split["param_min_samples_split"], result_samples_split["mean_test_score"])



ax21.set_xlabel("min_samples_leaf")

ax21.set_ylabel("f1 score")

ax21.plot(result_samples_leaf["param_min_samples_leaf"], result_samples_leaf["mean_test_score"])



ax22.set_xlabel("max_features")

ax22.plot(result_features["param_max_features"], result_features["mean_test_score"])
best_tree = DecisionTreeClassifier(max_depth = 6, min_samples_split = 11, min_samples_leaf = 3, max_features = 13)

best_tree.fit(X, y)
from sklearn.tree import export_graphviz



export_graphviz(best_tree, out_file='tree.dot', feature_names=X.columns)

print(open('tree.dot').read()) 
features = dict(zip(range(len(X.columns)), X.columns))



# Важность признаков

importances = best_tree.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = max(10, len(X.columns))

feature_indices = [ind for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features[feature_indices[f]], importances[indices[f]])



plt.figure(figsize=(15,5))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);
from sklearn.ensemble import RandomForestClassifier 

forest = RandomForestClassifier()

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

f1_score(y_pred, y_test)
params = {"n_estimators":[50, 100, 150, 200, 300, 400]}

forest_grid_estimators = GridSearchCV(forest, params, cv=kf, scoring='f1')

forest_grid_estimators.fit(X, y)

forest_grid_estimators.best_params_
params = {"max_depth":np.arange(2,21)}

forest = RandomForestClassifier(n_estimators = 300)

forest_grid_depth = GridSearchCV(tree, params, cv=kf, scoring='f1')

forest_grid_depth.fit(X, y)

forest_grid_depth.best_params_
params = {"min_samples_split":np.arange(2,21)}

forest = RandomForestClassifier(n_estimators = 300, max_depth = 10)

forest_grid_samples_split = GridSearchCV(tree, params, cv=kf, scoring='f1')

forest_grid_samples_split.fit(X, y)

forest_grid_samples_split.best_params_
params = {"min_samples_leaf":np.arange(2,21)}

forest = RandomForestClassifier(n_estimators = 300, max_depth= 10, min_samples_split = 8)

forest_grid_samples_leaf = GridSearchCV(tree, params, cv=kf, scoring='f1')

forest_grid_samples_leaf.fit(X, y)

forest_grid_samples_leaf.best_params_
params = {"max_features":np.arange(2,18)}

forest = RandomForestClassifier(n_estimators = 300, max_depth = 10, min_samples_split = 8, min_samples_leaf = 2)

forest_grid_features = GridSearchCV(tree, params, cv=kf, scoring='f1')

forest_grid_features.fit(X, y)

forest_grid_features.best_params_
result_estimators = pd.DataFrame(forest_grid_estimators.cv_results_)

result_depth = pd.DataFrame(forest_grid_depth.cv_results_)

result_samples_split = pd.DataFrame(forest_grid_samples_split.cv_results_)

result_samples_leaf = pd.DataFrame(forest_grid_samples_leaf.cv_results_)

result_features = pd.DataFrame(forest_grid_features.cv_results_)
import matplotlib.pyplot as plt

fig, ((ax11,ax12, ax13),(ax21,ax22, ax23)) = plt.subplots(2,3, figsize = (15,10))



ax11.set_xlabel("n_estimators")

ax11.set_ylabel("f1 score")

ax11.plot(result_estimators["param_n_estimators"], result_estimators["mean_test_score"])



ax12.set_xlabel("max_depth")

ax12.plot(results_depth["param_max_depth"], results_depth["mean_test_score"])



ax13.set_xlabel("min_samples_split")

ax13.plot(result_samples_split["param_min_samples_split"], result_samples_split["mean_test_score"])



ax21.set_xlabel("min_samples_leaf")

ax21.set_ylabel("f1 score")

ax21.plot(result_samples_leaf["param_min_samples_leaf"], result_samples_leaf["mean_test_score"])



ax22.set_xlabel("max_features")

ax22.plot(result_features["param_max_features"], result_features["mean_test_score"])
best_forest = RandomForestClassifier(n_estimators = 300, max_depth = 10, min_samples_split = 8, min_samples_leaf = 2, max_features = 16)
features = dict(zip(range(len(X.columns)), X.columns))



# Важность признаков

importances = best_tree.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = max(10, len(X.columns))

feature_indices = [ind for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features[feature_indices[f]], importances[indices[f]])



plt.figure(figsize=(15,5))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);
from sklearn.neighbors import KNeighborsClassifier

best_knn = KNeighborsClassifier(weights='distance', p = 1.49, n_neighbors=3)
from sklearn.model_selection import cross_val_score

print("KNN: ", cross_val_score(best_knn, X, y, cv = kf, scoring = "f1").mean())

print("Tree: ", cross_val_score(best_tree, X, y, cv = kf, scoring = "f1").mean())

print("Forest: ", cross_val_score(best_forest, X, y, cv = kf, scoring = "f1").mean())