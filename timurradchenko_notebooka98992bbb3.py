import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
adult_income_df = pd.read_csv('../input/adult-income-dataset/adult.csv')

adult_income_df.head().T
print((adult_income_df["workclass"] == "?").value_counts()[1])

print((adult_income_df["occupation"] == "?").value_counts()[1])

print((adult_income_df["native-country"] == "?").value_counts()[1])
adult_income_df = adult_income_df[adult_income_df["workclass"] != "?"]

adult_income_df = adult_income_df[adult_income_df["occupation"] != "?"]

adult_income_df = adult_income_df[adult_income_df["native-country"] != "?"]

                    

adult_income_df.shape
adult_income_df.replace(['Divorced', 'Married-AF-spouse', 

              'Married-civ-spouse', 'Married-spouse-absent', 

              'Never-married','Separated','Widowed'],

             ['Not Married','Married','Married','Married',

              'Not Married','Not Married','Not Married'], inplace = True)

adult_income_df.head().T
X = adult_income_df.drop('income', axis=1)

y = adult_income_df['income'].map({'<=50K':0, '>50K':1})

X.columns
num_features = ['age', 'fnlwgt', 'educational-num', 

                'capital-gain', 'capital-loss', 'hours-per-week']

X_num = X[num_features]

X_cat = X.drop(num_features, axis=1)
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()

X_cat_new = onehotencoder.fit_transform(X_cat).toarray()

X_names = np.hstack([num_features, onehotencoder.get_feature_names()])
X_new = np.hstack([X_num.values, X_cat_new])
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_scaled = sc.fit_transform(X_new)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=3, random_state=2019)

tree.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = tree.predict(X_test)

accuracy_score(y_test, y_pred)
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.model_selection import GridSearchCV

tree_params_depth = {'max_depth': np.arange(2, 11)}

tree_grid_depth = GridSearchCV(tree, tree_params_depth, cv=kf, scoring='accuracy') 

tree_grid_depth.fit(X_train, y_train)

print(tree_grid_depth.best_params_)

print(tree_grid_depth.best_score_)
tree = DecisionTreeClassifier(max_depth=tree_grid_depth.best_params_.get('max_depth'))

tree_params_split = {'min_samples_split': np.arange(2, 21)}

tree_grid_split = GridSearchCV(tree, tree_params_split, cv=kf, scoring='accuracy')

tree_grid_split.fit(X_train, y_train)

print(tree_grid_split.best_params_)

print(tree_grid_split.best_score_)
tree = DecisionTreeClassifier(max_depth=tree_grid_depth.best_params_.get('max_depth'), min_samples_split=tree_grid_split.best_params_.get('min_samples_split'))

tree_params_leaf = {'min_samples_leaf': np.arange(2, 21)}

tree_grid_leaf = GridSearchCV(tree, tree_params_leaf, cv=kf, scoring='accuracy')

tree_grid_leaf.fit(X_train, y_train)

print(tree_grid_leaf.best_params_)

print(tree_grid_leaf.best_score_)
tree = DecisionTreeClassifier(max_depth = tree_grid_depth.best_params_.get('max_depth'), min_samples_split = tree_grid_split.best_params_.get('min_samples_split'), min_samples_leaf = tree_grid_leaf.best_params_.get('min_samples_leaf'))

tree_params_features = {'max_features': np.arange(2, 51)}

tree_grid_features = GridSearchCV(tree, tree_params_features, cv=kf, scoring='accuracy')

tree_grid_features.fit(X_train, y_train)

print(tree_grid_features.best_params_)

print(tree_grid_features.best_score_)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize = (10,10))



ax[0,0].set_xlabel("Max depth")

ax[0,0].set_ylabel("Score")

ax[0,0].plot(tree_params_depth['max_depth'], tree_grid_depth.cv_results_["mean_test_score"]);



ax[0,1].set_xlabel("Min samples split")

ax[0,1].set_ylabel("Score")

ax[0,1].plot(tree_params_split["min_samples_split"], tree_grid_split.cv_results_["mean_test_score"]);



ax[1,0].set_xlabel("Min samples leaf")

ax[1,0].set_ylabel("Score")

ax[1,0].plot(tree_params_leaf["min_samples_leaf"],tree_grid_leaf.cv_results_["mean_test_score"]);



ax[1,1].set_xlabel("Max features")

ax[1,1].set_ylabel("Score")

ax[1,1].plot(tree_params_features["max_features"], tree_grid_features.cv_results_["mean_test_score"]);
optimal_tree = DecisionTreeClassifier(max_depth = tree_grid_depth.best_params_.get('max_depth'), min_samples_split = tree_grid_split.best_params_.get('min_samples_split'), min_samples_leaf = tree_grid_leaf.best_params_.get('min_samples_leaf'), max_features = tree_grid_features.best_params_.get('max_features'))

y_pred = optimal_tree.fit(X_train, y_train).predict(X_test)

accuracy_score(y_test, y_pred)
from sklearn.tree import export_graphviz

export_graphviz(optimal_tree, out_file='optimal_tree.dot', feature_names=X_names)

print(open('optimal_tree.dot').read())
features = dict(zip(range(len(X_names)), X_names))

importances = optimal_tree.feature_importances_

indices = np.argsort(importances)[::-1]

num_to_plot = len(X_names)

feature_indices = [ind for ind in indices[:num_to_plot]]



print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features[feature_indices[f]], importances[indices[f]])
plt.figure(figsize=(35,31))

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

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy_score(y_test, y_pred)
rf_params_estimators =  {'n_estimators': np.arange(100, 102)}

rf_grid_estimators = GridSearchCV(rf, rf_params_estimators, cv=kf, scoring='accuracy')

rf_grid_estimators.fit(X_train, y_train)

print(rf_grid_estimators.best_params_)

print(rf_grid_estimators.best_score_)
rf = RandomForestClassifier(n_estimators = rf_grid_estimators.best_params_.get('n_estimators'))

rf_params_depth = {'max_depth': np.arange(2, 11)}

rf_grid_depth = GridSearchCV(rf, rf_params_depth, cv=kf, scoring='accuracy')

rf_grid_depth.fit(X_train, y_train)

print(rf_grid_depth.best_params_)

print(rf_grid_depth.best_score_)
rf = RandomForestClassifier(n_estimators = rf_grid_estimators.best_params_.get('n_estimators'), max_depth = rf_grid_depth.best_params_.get('max_depth'))

rf_params_split = {'min_samples_split': np.arange(20, 31)}

rf_grid_split = GridSearchCV(rf, rf_params_split, cv=kf, scoring='accuracy')

rf_grid_split.fit(X_train, y_train)

print(rf_grid_split.best_params_)

print(rf_grid_split.best_score_)
rf = RandomForestClassifier(n_estimators = rf_grid_estimators.best_params_.get('n_estimators'), max_depth = rf_grid_depth.best_params_.get('max_depth'), min_samples_split =rf_grid_split.best_params_.get('min_samples_split'))

rf_params_leaf = {'min_samples_leaf': np.arange(18, 31)}

rf_grid_leaf = GridSearchCV(rf, rf_params_leaf, cv=kf, scoring='accuracy')

rf_grid_leaf.fit(X_train, y_train)

print(rf_grid_leaf.best_params_)

print(rf_grid_leaf.best_score_)
rf = RandomForestClassifier(n_estimators = rf_grid_estimators.best_params_.get('n_estimators'), max_depth = rf_grid_depth.best_params_.get('max_depth'), min_samples_split =rf_grid_split.best_params_.get('min_samples_split'), min_samples_leaf = rf_grid_leaf.best_params_.get('min_samples_leaf') )

rf_params_features = {'max_features': np.arange(18, 31)}

rf_grid_features = GridSearchCV(rf, rf_params_features, cv=kf, scoring='accuracy')

rf_grid_features.fit(X_train, y_train)

print(rf_grid_features.best_params_)

print(rf_grid_features.best_score_)
fig, ax = plt.subplots(2, 3, figsize = (15,15))



ax[0,0].set_xlabel("Max depth")

ax[0,0].set_ylabel("Score")

ax[0,0].plot(rf_params_depth['max_depth'], rf_grid_depth.cv_results_["mean_test_score"]);



ax[0,1].set_xlabel("Min samples split")

ax[0,1].set_ylabel("Score")

ax[0,1].plot(rf_params_split["min_samples_split"], rf_grid_split.cv_results_["mean_test_score"]);



ax[0,2].set_xlabel("Min samples leaf")

ax[0,2].set_ylabel("Score")

ax[0,2].plot(rf_params_leaf["min_samples_leaf"],rf_grid_leaf.cv_results_["mean_test_score"]);



ax[1,0].set_xlabel("Max features")

ax[1,0].set_ylabel("Score")

ax[1,0].plot(rf_params_features["max_features"], rf_grid_features.cv_results_["mean_test_score"]);



ax[1,1].set_xlabel("N Estimators")

ax[1,1].set_ylabel("Score")

ax[1,1].plot(rf_params_estimators["n_estimators"], rf_grid_estimators.cv_results_["mean_test_score"]);
optimal_rf = RandomForestClassifier(n_estimators = rf_grid_estimators.best_params_.get('n_estimators'), max_depth = rf_grid_depth.best_params_.get('max_depth'), min_samples_split =rf_grid_split.best_params_.get('min_samples_split'), min_samples_leaf = rf_grid_leaf.best_params_.get('min_samples_leaf') , max_features = rf_grid_features.best_params_.get('max_features'))

y_pred = optimal_rf.fit(X_train, y_train).predict(X_test)

accuracy_score(y_test, y_pred)
features = dict(zip(range(len(X_names)), X_names))

importances = optimal_rf.feature_importances_

indices = np.argsort(importances)[::-1]

num_to_plot = 10

feature_indices = [ind for ind in indices[:num_to_plot]]



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