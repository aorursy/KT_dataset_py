# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import seaborn as sns

sns.set(rc={'figure.figsize':(15, 10)});



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/adult-income-dataset/adult.csv")

df.head().T
sns.countplot(df['income'])
df1 = df.drop(['education','fnlwgt','native-country','relationship'], axis = 1)

print(df1.workclass.mode())

print(df1.occupation.mode())

df1['workclass'].replace({"?":"Other"}, inplace = True)

df1['occupation'].replace({"?":"Prof-specialty"}, inplace = True)

df1['income'] = df1['income'].map({'<=50K' : 0, '>50K' : 1})

df1['gender'] = df1['gender'].map({'Male' : 0, 'Female' : 1})

df1 = pd.get_dummies(df1)

df1.head().T
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df2 = df1.drop('income', axis=1) 

X = scaler.fit_transform(df2)
from sklearn.model_selection import train_test_split



y = df1['income'] 



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=11310)
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(max_depth=3, random_state=11310)

tree.fit(X_train, y_train)



from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix



y_pred = tree.predict(X_valid)

print(f1_score(y_valid, y_pred))



tn, fp, fn, tp = confusion_matrix(y_valid, y_pred).ravel()

print((tn,fp,fn,tp))
from sklearn.model_selection import GridSearchCV



tree_params = {'max_depth': np.arange(2, 45),

               #'min_samples_leaf': np.arange(2, 15),

               #'min_samples_split':np.arange(2,11),

               #'max_features':np.arange(2,11)

              }



tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='f1') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)

tree_grid.best_estimator_

tree_grid.best_score_

res = pd.DataFrame(tree_grid.cv_results_)

res.head().T

plt.plot(res['param_max_depth'], res['mean_test_score'])

plt.ylabel('f1 scoring')

plt.xlabel('max_depth')

plt.show()
tree_params = {'max_depth': [12],

               'min_samples_leaf': np.arange(2, 15),

               #min_samples_split:np.arange(2,11),

               #max_features:np.arange(2,11)

              }

tree_grid_1 = GridSearchCV(tree, tree_params, cv=5, scoring='f1') # кросс-валидация по 5 блокам

tree_grid_1.fit(X_train, y_train)
tree_grid_1.best_estimator_
tree_grid_1.best_score_
res = pd.DataFrame(tree_grid_1.cv_results_)

res.head().T

plt.plot(res['param_min_samples_leaf'], res['mean_test_score'])

plt.ylabel('f1 scoring')

plt.xlabel('min_samples_leaf')

plt.show()
tree_params = {'max_depth': [12],

               'min_samples_leaf': [8],

               'min_samples_split':np.arange(2,31)

               #max_features:np.arange(2,11)

              }

tree_grid_2 = GridSearchCV(tree, tree_params, cv=5, scoring='f1') # кросс-валидация по 5 блокам

tree_grid_2.fit(X_train, y_train)
tree_grid_2.best_params_
tree_grid_2.best_score_
res = pd.DataFrame(tree_grid_2.cv_results_)

res.head().T

plt.plot(res['param_min_samples_split'], res['mean_test_score'])

plt.ylabel('f1 scoring')

plt.xlabel('min_samples_split')

plt.show()
tree_params = {'max_depth': [12],

               'min_samples_leaf': [8],

               'min_samples_split':[2],

               'max_features':np.arange(2,41) #41 - кол-во фич

              }

tree_grid_3 = GridSearchCV(tree, tree_params, cv=5, scoring='f1') # кросс-валидация по 5 блокам

tree_grid_3.fit(X_train, y_train)
tree_grid_3.best_params_
tree_grid_3.best_score_
res = pd.DataFrame(tree_grid_3.cv_results_)

res.head().T

plt.plot(res['param_max_features'], res['mean_test_score'])

plt.ylabel('f1 scoring')

plt.xlabel('max_features')

plt.show()
tree = DecisionTreeClassifier(max_depth = 12, max_features = 35, min_samples_leaf = 8, min_samples_split = 2)

tree.fit(X_train, y_train)

from sklearn.tree import export_graphviz



export_graphviz(tree, out_file='tree.dot', feature_names=df2.columns)

print(open('tree.dot').read())
import matplotlib.pyplot as plt



features = dict(zip(range(len(df2.columns)-1), df2.columns[:-1]))



# Важность признаков

importances = tree.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = max(10, len(df.columns[:-1]))

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

df2['marital-status_Married-civ-spouse'].value_counts()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150, random_state=11335,max_depth=6)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)

f1_score(y_valid, y_pred)

rf_params = {'n_estimators':np.arange(10,500,25)

              }

rf_1 = GridSearchCV(rf, rf_params, cv=5, scoring='f1') # кросс-валидация по 5 блокам

rf_1.fit(X_train, y_train)



rf_1.best_params_
rf_1.best_score_
res = pd.DataFrame(rf_1.cv_results_)

res.head().T

plt.plot(res['param_n_estimators'], res['mean_test_score'])

plt.ylabel('f1 scoring')

plt.xlabel('Number of estimators')

plt.show()
rf_params = {'n_estimators':[60],

             'max_depth':np.arange(2,25)

              }

rf_2 = GridSearchCV(rf, rf_params, cv=5, scoring='f1') # кросс-валидация по 5 блокам

rf_2.fit(X_train, y_train)

rf_2.best_params_
rf_2.best_score_
res = pd.DataFrame(rf_2.cv_results_)

res.head().T

plt.plot(res['param_max_depth'], res['mean_test_score'])

plt.ylabel('f1 scoring')

plt.xlabel('Max depth')

plt.show()
rf_params = {'n_estimators':[60],

             'max_depth':[20],

             'min_samples_split':np.arange(2,25)

              }

rf_3 = GridSearchCV(rf, rf_params, cv=5, scoring='f1') # кросс-валидация по 5 блокам

rf_3.fit(X_train, y_train)

rf_3.best_params_
rf_3.best_score_
res = pd.DataFrame(rf_3.cv_results_)

res.head().T

plt.plot(res['param_min_samples_split'], res['mean_test_score'])

plt.ylabel('f1 scoring')

plt.xlabel('Min samples split')

plt.show()
rf_params = {'n_estimators':[60],

             'max_depth':[20],

             'min_samples_split':[17],

             'min_samples_leaf':np.arange(2,25)

              }

rf_4 = GridSearchCV(rf, rf_params, cv=5, scoring='f1') # кросс-валидация по 5 блокам

rf_4.fit(X_train, y_train)
rf_4.best_score_
rf_4.best_params_
res = pd.DataFrame(rf_4.cv_results_)

res.head().T

plt.plot(res['param_min_samples_leaf'], res['mean_test_score'])

plt.ylabel('f1 scoring')

plt.xlabel('Min samples leaf')

plt.show()
rf_params = {'n_estimators':[60],

             'max_depth':[20],

             'min_samples_split':[17],

             'min_samples_leaf':[2],

             'max_features':np.arange(2,41)

              }

rf_5 = GridSearchCV(rf, rf_params, cv=5, scoring='f1') # кросс-валидация по 5 блокам

rf_5.fit(X_train, y_train)
rf_5.best_params_
rf_5.best_score_
res = pd.DataFrame(rf_5.cv_results_)

res.head().T

plt.plot(res['param_max_features'], res['mean_test_score'])

plt.ylabel('f1 scoring')

plt.xlabel('Min samples split')

plt.show()
features = dict(zip(range(len(df2.columns)-1), df2.columns[:-1]))



# Важность признаков

importances = rf.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = max(10, len(df.columns[:-1]))

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