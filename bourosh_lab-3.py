# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/bank-marketing-dataset/bank.csv")

# Загружаем данные
df.info()
df.head()
df.describe()
sns.catplot(x="housing", kind="count", data=df)
df['deposit']
df1 = pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')
df1.info()
df1 = pd.get_dummies(df, columns=['job', 'education', 'marital', 'contact', 'poutcome', 'month', 'loan', 'housing', 'default'])

df1['deposit']=df['deposit'].map({'no':0,'yes':1})

df1.head()
df2 = df1.drop(['deposit'], axis=1)
df2.head()
from sklearn.model_selection import train_test_split



# Создание X, y

# X --- вся таблица без таргета

# y --- таргет (целевая переменная)

X=df1.drop('deposit', axis=1)

y =df1['deposit']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12)
df1.head().T
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                      test_size=0.3, random_state=2019)
# Обучение дерева решений

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(max_depth = 3, random_state = 2019)

tree.fit(X_train, y_train)

from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='tree.dot')

print(open('tree.dot').read()) 
from sklearn.metrics import accuracy_score

y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
from sklearn.model_selection import GridSearchCV

tree_params = {'max_depth': np.arange(2, 11),

               'min_samples_leaf': np.arange(2, 11)}



tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)
best_tree = tree_grid.best_estimator_

y_pred = best_tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
pd.DataFrame(tree_grid.cv_results_).T


tree_params = {'max_depth': np.arange(2, 11),

               #'min_samples_leaf': np.arange(2, 11),

               #'min_samples_split':np.arange(2,11),

               #'max_features':np.arange(2,11)

              }



tree_grid_max_depth = GridSearchCV(tree, tree_params, cv=kf, scoring='accuracy')

tree_grid_max_depth.fit(X_train, y_train)
end = pd.DataFrame(tree_grid_max_depth.cv_results_)

end.head().T

plt.plot(end['param_max_depth'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('max_depth')

plt.figure()
tree_grid_max_depth.best_params_
tree_grid_max_depth.best_estimator_
tree_grid_max_depth.best_score_
tree_params = {'max_depth': [9],

               'min_samples_leaf': np.arange(2, 11),

               #'min_samples_split':np.arange(2,11),

               #'max_features':np.arange(2,11)

              }

tree_grid_min_samples_leaf = GridSearchCV(tree, tree_params, cv=kf, scoring='accuracy')

tree_grid_min_samples_leaf.fit(X_train, y_train)
end = pd.DataFrame(tree_grid_min_samples_leaf.cv_results_)

end.head().T

plt.plot(end['param_min_samples_leaf'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('min_samples_leaf')

plt.figure()
tree_grid_min_samples_leaf.best_score_
tree_grid_min_samples_leaf.best_estimator_
tree_grid_min_samples_leaf.best_params_
tree_params = {'max_depth': [9],

               'min_samples_leaf': [3],

                'min_samples_split':np.arange(2,21)

              }

tree_grid_min_samples_split = GridSearchCV(tree, tree_params, cv=kf, scoring='accuracy') 

tree_grid_min_samples_split.fit(X_train, y_train)
end = pd.DataFrame(tree_grid_min_samples_split.cv_results_)

end.head()

plt.plot(end['param_min_samples_split'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('min_samples_split')

plt.figure()
tree_grid_min_samples_split.best_params_
tree_grid_min_samples_split.best_estimator_
tree_grid_min_samples_split.best_score_
tree_params = {'max_depth': [9],

               'min_samples_leaf': [3],

               'min_samples_split': [11],

               'max_features': np.arange(1, X.shape[1])

              }

tree_grid_max_features = GridSearchCV(tree, tree_params, cv=kf, scoring='accuracy') 

tree_grid_max_features.fit(X_train, y_train)
end = pd.DataFrame(tree_grid_max_features.cv_results_)

end.head().T

plt.plot(end['param_max_features'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('max_features')

plt.figure()
tree_grid_max_features.best_params_
tree_grid_max_features.best_estimator_
tree_grid_max_features.best_score_
best_tree_max_depth = tree_grid_max_depth.best_estimator_

y_pred = best_tree_max_depth.predict(X_valid)

accuracy_score(y_valid, y_pred)
export_graphviz(best_tree_max_depth, out_file='tree.dot')

print(open('tree.dot').read()) 
from sklearn.ensemble import RandomForestClassifier

tree_params = {'n_estimators': np.arange(165, 196, 10), 'max_depth': np.arange(6, 13)}

tree = RandomForestClassifier(random_state=22)

tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy')

tree_grid.fit(X_train, y_train)



print(tree_grid.best_score_)

print(tree_grid.best_params_)

#print(tree_grid.best_estimator_)


features = dict(zip(range(len(X.columns)), X.columns))



# Важность признаков

importances = tree_grid.best_estimator_.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = min(10, len(X.columns))

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

plt.legend(bars, [u''.join(features[i]) for i in feature_indices])

plt.show()

rf = RandomForestClassifier(n_estimators=100, random_state=2020)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)



accuracy_score(y_valid, y_pred)
rf = RandomForestClassifier(n_estimators=100, random_state=2020, max_depth=4)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)



accuracy_score(y_valid, y_pred)
rf_params_n_estimators = {'n_estimators': np.arange(5, 100, 10)}

rf_n_estimators = RandomForestClassifier(random_state=2020)

rf_grid_n_estimators = GridSearchCV(rf_n_estimators, rf_params_n_estimators, cv=5, scoring='accuracy') 

rf_grid_n_estimators.fit(X_train, y_train)
print(rf_grid_n_estimators.best_score_)

print(rf_grid_n_estimators.best_params_)

print(rf_grid_n_estimators.best_estimator_)
import matplotlib.pyplot as plt



fig, ax = plt.subplots()



ax.plot(rf_params_n_estimators['n_estimators'], rf_grid_n_estimators.cv_results_['mean_test_score']) # accuracy vs max_depth

ax.set_xlabel('n_estimators')

ax.set_ylabel('Mean accuracy on test set')
rf_params_max_features = {'max_features': np.arange(5, 100, 10)}

rf_max_features = RandomForestClassifier(n_estimators=95, random_state=2020)

rf_grid_max_features = GridSearchCV(rf_max_features, rf_params_max_features, cv=5, scoring='accuracy')

rf_grid_max_features.fit(X_train, y_train)
print(rf_grid_max_features.best_score_)

print(rf_grid_max_features.best_params_)

print(rf_grid_max_features.best_estimator_)
fig, ax = plt.subplots() 



ax.plot(rf_params_max_features['max_features'], rf_grid_max_features.cv_results_['mean_test_score']) # accuracy vs max_depth

ax.set_xlabel('max_features')

ax.set_ylabel('Mean accuracy on test set')
rf_params_max_depth = {'max_depth': np.arange(2, 11)}

rf_max_depth = RandomForestClassifier(n_estimators=95, max_features=45, random_state=2020)

rf_grid_max_depth = GridSearchCV(rf_max_depth, rf_params_max_depth, cv=5, scoring='accuracy') 

rf_grid_max_depth.fit(X_train, y_train)
print(rf_grid_max_depth.best_score_)

print(rf_grid_max_depth.best_params_)

print(rf_grid_max_depth.best_estimator_)
fig, ax = plt.subplots() 



ax.plot(rf_params_max_depth['max_depth'], rf_grid_max_depth.cv_results_['mean_test_score']) # accuracy vs max_depth

ax.set_xlabel('max_depth')

ax.set_ylabel('Mean accuracy on test set')
rf_params_min_samples_leaf = {'min_samples_leaf': np.arange(3, 10, 2)}

rf_min_samples_leaf = RandomForestClassifier(n_estimators=95, max_features=45, max_depth=10, random_state=22)

rf_grid_min_samples_leaf = GridSearchCV(rf_min_samples_leaf, rf_params_min_samples_leaf, cv=5, scoring='accuracy')

rf_grid_min_samples_leaf.fit(X_train, y_train)
print(rf_grid_min_samples_leaf.best_score_)

print(rf_grid_min_samples_leaf.best_params_)

print(rf_grid_min_samples_leaf.best_estimator_)
fig, ax = plt.subplots() 



ax.plot(rf_params_min_samples_leaf['min_samples_leaf'], rf_grid_min_samples_leaf.cv_results_['mean_test_score']) # accuracy vs max_depth

ax.set_xlabel('min_samples_leaf')

ax.set_ylabel('Mean accuracy on test set')
rf_params = {'n_estimators': np.arange(165, 196, 10), 'max_depth': np.arange(6, 13)}

rf = RandomForestClassifier(random_state=22)

rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy')

rf_grid.fit(X_train, y_train)



print(rf_grid.best_score_)

print(rf_grid.best_params_)

#print(rf_grid.best_estimator_)
features = dict(zip(range(len(X.columns)), X.columns))



# Важность признаков

importances = rf_grid.best_estimator_.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = min(10, len(X.columns))

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

plt.legend(bars, [u''.join(features[i]) for i in feature_indices])

plt.show()
