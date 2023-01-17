import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); 
df = pd.read_csv('../input/adult-income-dataset/adult.csv')
df.head(10).T
df.info()
df.describe().T
df['income'].value_counts().plot(kind='bar', color='black', figsize=(5,5))

plt.figure()

df['income']=df['income'].map({ '>50K': 1, '<=50K': 0})

df.head(10).T
new_df = pd.get_dummies(df, columns=['workclass', 'occupation', 'native-country']) 

new_df.head()
df = df[df["workclass"] != "?"]

df = df[df["occupation"] != "?"]

df = df[df["native-country"] != "?"]

df.head()
df1 =['workclass', 'race', 'education','marital-status', 'occupation', 'relationship', 'gender','native-country',

      'income'] 

for i in df1:

    unique_value, index = np.unique(df[i], return_inverse=True) 

    df[i] = index

df.head(10).T
from sklearn.model_selection import train_test_split

X = df.drop(['income'], axis=1)

y = df['income']

X_train, X_valid, y_train, y_valid = train_test_split(X, y,

                                                      test_size=0.3, random_state=2019) # random_state=2019 
# Обучение дерева решений

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(max_depth=3, random_state=2019)

tree.fit(X_train, y_train)
# Предсказания для валидационного множества

from sklearn.metrics import accuracy_score



y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import GridSearchCV



tree_params = {'max_depth': np.arange(2, 11),

               #'min_samples_leaf': np.arange(2, 11),

               #'min_samples_split':np.arange(2,11),

               #'max_features':np.arange(2,11)

              }



tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid.fit(X_train, y_train)
tree_grid.best_estimator_
tree_grid.best_score_
end = pd.DataFrame(tree_grid.cv_results_)

end.head().T

plt.plot(end['param_max_depth'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('max_depth')

plt.figure()
tree_params = {'max_depth': [9],

               'min_samples_leaf': np.arange(2, 11),

               #'min_samples_split':np.arange(2,11),

               #'max_features':np.arange(2,11)

              }

tree_grid_1 = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid_1.fit(X_train, y_train)
tree_grid_1.best_estimator_
tree_grid_1.best_score_
end = pd.DataFrame(tree_grid_1.cv_results_)

end.head().T

plt.plot(end['param_min_samples_leaf'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('min_samples_leaf')

plt.figure()
tree_params = {'max_depth': [9],

               'min_samples_leaf': [3],

                'min_samples_split':np.arange(2,21)

              }

tree_grid_2 = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') 

tree_grid_2.fit(X_train, y_train)
tree_grid_2.best_params_
tree_grid_2.best_score_
end = pd.DataFrame(tree_grid_2.cv_results_)

end.head().T

plt.plot(end['param_min_samples_split'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('min_samples_split')

plt.figure()
tree_params = {'max_depth': [9],

               'min_samples_leaf': [3],

               'min_samples_split': [11],

               'max_features': np.arange(1, X.shape[1])

              }

tree_grid_3 = GridSearchCV(tree, tree_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

tree_grid_3.fit(X_train, y_train)
tree_grid_3.best_params_
tree_grid_3.best_score_
end = pd.DataFrame(tree_grid_3.cv_results_)

end.head().T

plt.plot(end['param_max_features'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('max_features')

plt.figure()
best_tree = DecisionTreeClassifier(max_depth = 9, max_features = 13, min_samples_leaf = 3, min_samples_split = 11)

best_tree.fit(X_train, y_train)
from sklearn.tree import export_graphviz

export_graphviz(best_tree, out_file='best_tree.dot')

print(open('best_tree.dot').read())
features = dict(zip(range(len(X.columns)), X.columns))



# Importance of features

importances = best_tree.feature_importances_

indices = np.argsort(importances)[::-1]



# Plot the feature importancies of the forest

num_to_plot = max(10, len(X.columns))

feature_indices = [ind for ind in indices[:num_to_plot]]



print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features[feature_indices[f]], importances[indices[f]])
plt.figure(figsize=(20,10))

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



#rf = RandomForestClassifier(n_estimators=100, random_state=2019, max_depth=6)



rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)



accuracy_score(y_valid, y_pred)
rf_param = {'n_estimators': [50, 100, 200, 300]

           }

new_rf = GridSearchCV(rf, rf_param, cv=5, scoring='accuracy')

new_rf.fit(X_train, y_train)
new_rf.best_params_
new_rf.best_score_
end = pd.DataFrame(new_rf.cv_results_)

end.head().T

plt.plot(end['param_n_estimators'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('Number of estimators')

plt.figure()
rf_params = {'n_estimators':[200],

             'max_depth':np.arange(2,25)

              }

new_rf_2 = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

new_rf_2.fit(X_train, y_train)
new_rf_2.best_params_
new_rf_2.best_score_
end = pd.DataFrame(new_rf_2.cv_results_)

end.head().T

plt.plot(end['param_max_depth'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('Max depth')

plt.figure()
rf_params = {'n_estimators':[200],

             'max_depth':[17],

             'min_samples_split':np.arange(2,25)

              }

new_rf_3 = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

new_rf_3.fit(X_train, y_train)
new_rf_3.best_params_
new_rf_3.best_score_
end = pd.DataFrame(new_rf_3.cv_results_)

end.head().T

plt.plot(end['param_min_samples_split'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('Min samples split')

plt.figure()
rf_params = {'n_estimators':[200],

             'max_depth':[17],

             'min_samples_split':[22],

             'min_samples_leaf':np.arange(2,25)

              }

new_rf_4 = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

new_rf_4.fit(X_train, y_train)
new_rf_4.best_params_
new_rf_4.best_score_
end = pd.DataFrame(new_rf_4.cv_results_)

end.head().T

plt.plot(end['param_min_samples_leaf'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('Min samples leaf')

plt.figure()
rf_params = {'n_estimators':[200],

             'max_depth':[17],

             'min_samples_split':[22],

             'min_samples_leaf':[2],

             'max_features': np.arange(1, X.shape[1])

              }

new_rf_5 = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy') # кросс-валидация по 5 блокам

new_rf_5.fit(X_train, y_train)

new_rf_5.best_params_
new_rf_5.best_score_
end = pd.DataFrame(new_rf_5.cv_results_)

end.head().T

plt.plot(end['param_max_features'], end['mean_test_score'])

plt.ylabel('accuracy scoring')

plt.xlabel('Min samples split')

plt.figure()
plt.figure(figsize=(20,10))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")



ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)



plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);
