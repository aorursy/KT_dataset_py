import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
df = pd.read_csv('../input/labanumbatree/bigml_59c28831336c6604c800002a.csv')

df.head()
df.describe()
df.info()
sns.countplot(df['churn'])
sns.boxplot(df['total eve charge'])
sns.boxplot(df['account length'])
sns.boxplot(df['total day minutes'])
sns.boxplot(df['total night minutes'])
sns.boxplot(df['total intl minutes'])
ndf=pd.get_dummies(df,columns=['state','international plan','voice mail plan'])

ndf=df.copy()

from sklearn. preprocessing import LabelEncoder



le = LabelEncoder()

le.fit(df['state'])

ndf['state']=le.transform(df['state'])



le = LabelEncoder()

le.fit(df['international plan'])

ndf['international plan']=le.transform(df['international plan'])



le = LabelEncoder()

le.fit(df['voice mail plan'])

ndf['voice mail plan']=le.transform(df['voice mail plan'])
ndf=ndf.drop('phone number',axis=1)
ndf=ndf.drop('area code', axis=1)
ndf['account length']=ndf['account length']/365
ndf['churn']=ndf['churn'].map({True:1,False:0})
ndf.head()
ndf.info()
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()



X1=ndf.drop('churn',axis=1)

print(X1)
X=scaler.fit_transform(X1)

print(X)
y=ndf['churn']

print(y)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=2019)

tree.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

kf = KFold(n_splits=5, shuffle=True, random_state=42)
tree = DecisionTreeClassifier(random_state=2019)

tree_p_max_depth = {'max_depth': np.arange(2, 20)}

tree_grid_max_depth = GridSearchCV(tree, tree_p_max_depth, cv=kf, scoring='accuracy')

tree_grid_max_depth.fit(X_train, y_train)
tree_grid_max_depth.best_estimator_
tree_grid_max_depth.best_score_
pd.DataFrame(tree_grid_max_depth.cv_results_).T
tree = DecisionTreeClassifier(random_state=2019,max_depth=6)

tree_p_min_samples_split = {'min_samples_split': np.arange(2, 20)}

tree_grid_min_samples_split = GridSearchCV(tree, tree_p_min_samples_split, cv=kf, scoring='accuracy')

tree_grid_min_samples_split.fit(X_train, y_train)
tree_grid_min_samples_split.best_estimator_
tree_grid_min_samples_split.best_score_
pd.DataFrame(tree_grid_min_samples_split.cv_results_).T
tree = DecisionTreeClassifier(random_state=2019,max_depth=6,min_samples_split=14)

tree_p_min_samples_leaf = {'min_samples_leaf': np.arange(1, 20)}

tree_grid_min_samples_leaf = GridSearchCV(tree, tree_p_min_samples_leaf, cv=kf, scoring='accuracy')

tree_grid_min_samples_leaf.fit(X_train, y_train)
tree_grid_min_samples_leaf.best_estimator_
tree_grid_min_samples_leaf.best_score_
pd.DataFrame(tree_grid_min_samples_leaf.cv_results_).T
tree = DecisionTreeClassifier(random_state=2019,max_depth=6,min_samples_split=14,min_samples_leaf=2)

tree_p_max_features = {'max_features': np.arange(1, X.shape[1])}

tree_grid_max_features = GridSearchCV(tree, tree_p_max_features, cv=kf, scoring='accuracy')

tree_grid_max_features.fit(X_train, y_train)
tree_grid_max_features.best_estimator_
tree_grid_max_features.best_score_
pd.DataFrame(tree_grid_max_features.cv_results_).T
fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(30,30))



ax[0, 0].plot(tree_p_max_depth['max_depth'], tree_grid_max_depth.cv_results_['mean_test_score'])

ax[0, 0].set_xlabel('max_depth')

ax[0, 0].set_ylabel('Mean accuracy on test set')



ax[0, 1].plot(tree_p_min_samples_split['min_samples_split'], tree_grid_min_samples_split.cv_results_['mean_test_score'])

ax[0, 1].set_xlabel('min_samples_split')

ax[0, 1].set_ylabel('Mean accuracy on test set')



ax[1, 0].plot(tree_p_min_samples_leaf['min_samples_leaf'], tree_grid_min_samples_leaf.cv_results_['mean_test_score'])

ax[1, 0].set_xlabel('min_samples_leaf')

ax[1, 0].set_ylabel('Mean accuracy on test set')



ax[1, 1].plot(tree_p_max_features['max_features'], tree_grid_max_features.cv_results_['mean_test_score'])

ax[1, 1].set_xlabel('max_features')

ax[1, 1].set_ylabel('Mean accuracy on test set')
best_tree=DecisionTreeClassifier(max_depth=6, max_features=17, min_samples_leaf=2,

                       min_samples_split=14, random_state=2019)
y_pred =best_tree.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.tree import export_graphviz

export_graphviz(best_tree, out_file='best_tree.dot', feature_names=X1.columns)

print(open('best_tree.dot').read())
features = {'f'+str(i+1):name for (i, name) in zip(range(len(ndf.columns)), ndf.columns)}



importances = best_tree.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the tree

num_to_plot = 10

feature_indices = [ind+1 for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features["f"+str(feature_indices[f])], importances[indices[f]])



plt.figure(figsize=(20,10))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features["f"+str(i)]) for i in feature_indices]);
from scipy.stats import pointbiserialr

pointbiserialr(ndf['total day charge'], ndf['churn'])
pointbiserialr(ndf['customer service calls'], ndf['churn'])
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state = 2019)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)

accuracy_score(y_valid, y_pred)
rf = RandomForestClassifier(random_state=2019)

rf_p_n_estimators = {'n_estimators': np.arange(50, 501,50)}

rf_grid_n_estimators = GridSearchCV(rf, rf_p_n_estimators, cv=kf, scoring='accuracy')

rf_grid_n_estimators.fit(X_train, y_train)
rf_grid_n_estimators.best_estimator_
rf_grid_n_estimators.best_score_
pd.DataFrame(rf_grid_n_estimators.cv_results_).T
rf = RandomForestClassifier(random_state=2019,n_estimators=150)

rf_p_max_depth = {'max_depth': np.arange(2, 20)}

rf_grid_max_depth = GridSearchCV(rf, rf_p_max_depth, cv=kf, scoring='accuracy')

rf_grid_max_depth.fit(X_train, y_train)
rf_grid_max_depth.best_estimator_
rf_grid_max_depth.best_score_
pd.DataFrame(rf_grid_max_depth.cv_results_).T
rf = RandomForestClassifier(random_state=2019,n_estimators=150,max_depth=14)

rf_p_min_samples_split = {'min_samples_split': np.arange(2, 30)}

rf_grid_min_samples_split = GridSearchCV(rf, rf_p_min_samples_split, cv=kf, scoring='accuracy')

rf_grid_min_samples_split.fit(X_train, y_train)
rf_grid_min_samples_split.best_estimator_
rf_grid_min_samples_split.best_score_
pd.DataFrame(rf_grid_min_samples_split.cv_results_).T
rf = RandomForestClassifier(random_state=2019,n_estimators=150,max_depth=14,min_samples_split=12)

rf_p_min_samples_leaf = {'min_samples_leaf': np.arange(1, 20)}

rf_grid_min_samples_leaf = GridSearchCV(rf, rf_p_min_samples_leaf, cv=kf, scoring='accuracy')

rf_grid_min_samples_leaf.fit(X_train, y_train)
rf_grid_min_samples_leaf.best_estimator_
rf_grid_min_samples_leaf.best_score_
pd.DataFrame(rf_grid_min_samples_leaf.cv_results_).T
rf = RandomForestClassifier(random_state=2019,n_estimators=150,max_depth=14,min_samples_split=12,min_samples_leaf=1)

rf_p_max_features = {'max_features': np.arange(1, X.shape[1])}

rf_grid_max_features = GridSearchCV(rf, rf_p_max_features, cv=kf, scoring='accuracy')

rf_grid_max_features.fit(X_train, y_train)
rf_grid_max_features.best_estimator_
rf_grid_max_features.best_score_
pd.DataFrame(rf_grid_max_features.cv_results_).T
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=5, sharey=True,figsize=(30, 6))





ax[0].plot(rf_p_n_estimators['n_estimators'], rf_grid_n_estimators.cv_results_['mean_test_score'])

ax[0].set_xlabel('n_estimators')

ax[0].set_ylabel('Mean accuracy on test set')



ax[1].plot(rf_p_max_depth['max_depth'], rf_grid_max_depth.cv_results_['mean_test_score'])

ax[1].set_xlabel('max_depth')

ax[1].set_ylabel('Mean accuracy on test set')



ax[2].plot(rf_p_min_samples_split['min_samples_split'], rf_grid_min_samples_split.cv_results_['mean_test_score'])

ax[2].set_xlabel('min_samples_split')

ax[2].set_ylabel('Mean accuracy on test set')



ax[3].plot(rf_p_min_samples_leaf['min_samples_leaf'], rf_grid_min_samples_leaf.cv_results_['mean_test_score'])

ax[3].set_xlabel('min_samples_leaf')

ax[3].set_ylabel('Mean accuracy on test set')



ax[4].plot(rf_p_max_features['max_features'], rf_grid_max_features.cv_results_['mean_test_score'])

ax[4].set_xlabel('max_features')

ax[4].set_ylabel('Mean accuracy on test set')
best_rf = RandomForestClassifier(n_estimators=150, max_depth=14,min_samples_split=12,max_features=5)# Гиперпараметр min_samples_leaf=1 стоит по умолчанию, можно не указывать

y_pred =best_rf.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
features = {'f'+str(i+1):name for (i, name) in zip(range(len(ndf.columns)), ndf.columns)}



importances = best_rf.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the random forest

num_to_plot = 10

feature_indices = [ind+1 for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features["f"+str(feature_indices[f])], importances[indices[f]])



plt.figure(figsize=(20,10))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features["f"+str(i)]) for i in feature_indices]);
best_rf.score(X_valid,y_valid)