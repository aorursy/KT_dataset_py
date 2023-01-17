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
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
y = df['quality'] 

X = df.drop('quality', axis=1)



X.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                      test_size=0.3, random_state=2019)
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(max_depth=5, random_state=2019)

tree.fit(X_train, y_train)
from sklearn.metrics import accuracy_score



y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
# Визуализация

from sklearn.tree import export_graphviz



export_graphviz(tree, out_file='tree.dot', feature_names=X.columns)

print(open('tree.dot').read()) 
# Кросс-валидация и подбор гиперпараметров

from sklearn.model_selection import GridSearchCV



tree_params = {'max_depth': np.arange(2, 11),

               'min_samples_leaf': np.arange(2, 11),

               'max_features': np.arange(2, 11)}



tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)
tree_grid.best_score_

tree_grid.best_params_
# Кросс-валидация и подбор гиперпараметров

from sklearn.model_selection import GridSearchCV



tree_params = {'max_depth': np.arange(8, 13),

               'min_samples_leaf': np.arange(9, 15),

               'max_features': np.arange(4, 7)}



tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)
tree_grid.best_score_
pd.DataFrame(tree_grid.cv_results_).head().T



best_tree = tree_grid.best_estimator_

y_pred = best_tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=200, random_state=2019, max_depth=9)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_valid)



accuracy_score(y_valid, y_pred)
rf_params = {'max_depth': np.array([6, 9]),

            'n_estimators' : np.array([50, 100, 200])}



rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1) # кросс-валидация по 5 блокам

rf_grid.fit(X_train, y_train)
rf_grid.best_score_
rf_grid.best_params_
import matplotlib.pyplot as plt



features = {'f'+str(i+1):name for (i, name) in zip(range(len(df.columns)), df.columns)}



# Важность признаков



from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=200, random_state=2019, max_depth=9)

forest.fit(X_train, y_train)



importances = forest.feature_importances_



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