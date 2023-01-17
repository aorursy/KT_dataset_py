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
import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv("../input/../input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv", sep=",")

df.T
df.groupby("churn")["phone number"].count().plot(kind='bar')
df["state"].value_counts().shape[0]
df["area code"].value_counts()
df1 = df.drop(["state", "phone number"], axis=1)

df2 = df1.copy()



df2["area code"] = df['area code'].map({408:0,415:1,510:2})

df2["international plan"] = df['international plan'].map({"no":0,"yes":1})

df2["voice mail plan"] = df['voice mail plan'].map({"no":0,"yes":1})

df2["churn"] = df['churn'].map({False:0,True:1})



df2.head()
df_without_target = df2.drop(["churn"], axis=1)

df_without_target.head() 
X = df_without_target

X.head()
from sklearn.model_selection import train_test_split



y = df2["churn"]



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state=2019)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score



tree = DecisionTreeClassifier(max_depth=3, random_state=2019)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_valid)

f1_score(y_valid, y_pred)
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
from sklearn.model_selection import GridSearchCV



tree_params = {'max_depth': np.arange(2, 20)}



tree_grid_depth = GridSearchCV(tree, tree_params, cv=kf, scoring='f1')

tree_grid_depth.fit(X_train, y_train)
tree_grid_depth.best_params_
tree_grid_depth.best_score_
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)



ax[0].plot(tree_params['max_depth'], tree_grid_depth.cv_results_['mean_test_score']) 

ax[0].set_xlabel('max_depth')

ax[0].set_ylabel('f1_score')
tree = DecisionTreeClassifier(max_depth=6, random_state=2019)

tree_params = {'min_samples_split': np.arange(2, 20)}



tree_grid_split = GridSearchCV(tree, tree_params, cv=kf, scoring='f1')

tree_grid_split.fit(X_train, y_train)
tree_grid_split.best_params_
tree_grid_split.best_score_
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)



ax[0].plot(tree_params['min_samples_split'], tree_grid_split.cv_results_['mean_test_score']) 

ax[0].set_xlabel('min_samples_split')

ax[0].set_ylabel('f1_score')
tree = DecisionTreeClassifier(max_depth=8, min_samples_split=2, random_state=2019)

tree_params = {'min_samples_leaf': np.arange(2, 20)}



tree_grid_leaf = GridSearchCV(tree, tree_params, cv=kf, scoring='f1')

tree_grid_leaf.fit(X_train, y_train)
tree_grid_leaf.best_params_
tree_grid_leaf.best_score_
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)



ax[0].plot(tree_params['min_samples_leaf'], tree_grid_leaf.cv_results_['mean_test_score']) 

ax[0].set_xlabel('min_samples_leaf')

ax[0].set_ylabel('f1_score')
tree = DecisionTreeClassifier(max_depth=8, min_samples_split=2, min_samples_leaf=16, random_state=2019)

tree_params = {'max_features': np.arange(2, 18)}



tree_grid_features = GridSearchCV(tree, tree_params, cv=kf, scoring='f1')

tree_grid_features.fit(X_train, y_train)
tree_grid_features.best_params_
tree_grid_features.best_score_
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)



ax[0].plot(tree_params['max_features'], tree_grid_features.cv_results_['mean_test_score']) 

ax[0].set_xlabel('max_features')

ax[0].set_ylabel('f1_score')
best_tree = DecisionTreeClassifier(max_depth = 6, min_samples_split = 2, min_samples_leaf = 16, max_features = 13)

best_tree.fit(X_train, y_train)

mock_best_tree = DecisionTreeClassifier(max_depth = 3, min_samples_split = 2, min_samples_leaf = 16, max_features = 13)

mock_best_tree.fit(X_train, y_train)
from sklearn.tree import export_graphviz



export_graphviz(mock_best_tree, out_file='tree.dot', feature_names=X.columns)

print(open('tree.dot').read()) 
import matplotlib.pyplot as plt



features = {'f'+str(i+1):name for (i, name) in zip(range(len(df_without_target.columns)), df_without_target.columns)}



#Важность признаков



importances = best_tree.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = 10

feature_indices = [ind+1 for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features["f"+str(feature_indices[f])], importances[indices[f]])



plt.figure(figsize=(15,5))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features["f"+str(i)]) for i in feature_indices]);
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=2019)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)



f1_score(y_valid, y_pred)
rf_params = {"n_estimators":[50, 100, 150, 200, 300, 400, 450]}

rf_grid_estimators = GridSearchCV(rf, rf_params, cv=kf, scoring='f1')

rf_grid_estimators.fit(X_train, y_train)
rf_grid_estimators.best_params_
rf_grid_estimators.best_score_
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)



ax[0].plot(rf_params['n_estimators'], rf_grid_estimators.cv_results_['mean_test_score']) 

ax[0].set_xlabel('n_estimators')

ax[0].set_ylabel('f1_score')
rf = RandomForestClassifier(n_estimators=400, random_state=2019)

rf_params = {'max_depth': np.arange(2, 20)}

rf_grid_depth = GridSearchCV(rf, rf_params, cv=kf, scoring='f1')

rf_grid_depth.fit(X_train, y_train)
rf_grid_depth.best_params_
rf_grid_depth.best_score_
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)



ax[0].plot(rf_params['max_depth'], rf_grid_depth.cv_results_['mean_test_score']) 

ax[0].set_xlabel('max_depth')

ax[0].set_ylabel('f1_score')
rf = RandomForestClassifier(n_estimators=400, max_depth=16, random_state=2019)

rf_params = {'min_samples_split': np.arange(2, 20)}

rf_grid_split = GridSearchCV(rf, rf_params, cv=kf, scoring='f1')

rf_grid_split.fit(X_train, y_train)
rf_grid_split.best_params_
rf_grid_split.best_score_
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)



ax[0].plot(rf_params['min_samples_split'], rf_grid_split.cv_results_['mean_test_score']) 

ax[0].set_xlabel('min_samples_split')

ax[0].set_ylabel('f1_score')
rf = RandomForestClassifier(n_estimators=400, max_depth=16, min_samples_split=8, random_state=2019)

rf_params = {'min_samples_leaf': np.arange(2, 20)}

rf_grid_leaf = GridSearchCV(rf, rf_params, cv=kf, scoring='f1')

rf_grid_leaf.fit(X_train, y_train)
rf_grid_leaf.best_params_
rf_grid_leaf.best_score_
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)



ax[0].plot(rf_params['min_samples_leaf'], rf_grid_leaf.cv_results_['mean_test_score']) 

ax[0].set_xlabel('min_samples_leaf')

ax[0].set_ylabel('f1_score')
rf = RandomForestClassifier(n_estimators=400, max_depth=16, min_samples_split=8, min_samples_leaf=2, random_state=2019)

rf_params = {'max_features': np.arange(2, 18)}

rf_grid_features = GridSearchCV(rf, rf_params, cv=kf, scoring='f1')

rf_grid_features.fit(X_train, y_train)
rf_grid_features.best_params_
rf_grid_features.best_score_
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)



ax[0].plot(rf_params['max_features'], rf_grid_features.cv_results_['mean_test_score']) 

ax[0].set_xlabel('max_features')

ax[0].set_ylabel('f1_score')
best_forest = RandomForestClassifier(n_estimators=400, max_depth=16, min_samples_split=8, min_samples_leaf=2, max_features=11)

best_forest.fit(X_train, y_train)
import matplotlib.pyplot as plt



features = {'f'+str(i+1):name for (i, name) in zip(range(len(df_without_target.columns)), df_without_target.columns)}



#Важность признаков



importances = best_forest.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = 10

feature_indices = [ind+1 for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features["f"+str(feature_indices[f])], importances[indices[f]])



plt.figure(figsize=(15,5))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features["f"+str(i)]) for i in feature_indices]);