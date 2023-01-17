import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set();
df= pd.read_csv('../input/bank-marketing-dataset/bank.csv')
df.head(10)
df.info()
df.describe().T
def outliers_indices(feature):

    mid = df[feature].mean()

    sigma = df[feature].std()

    return df[(df[feature] < mid - 3*sigma) | (df[feature] > mid + 3*sigma)].index
wrong_dur=outliers_indices('duration')

wrong_bal=outliers_indices('balance')

out=set(wrong_bal|wrong_dur)

len(out)
df.info()
df.drop(out, inplace=True)
df.head()
# ddf= df.copy()
df['deposit']=df['deposit'].map({'no': 0,'yes': 1})
#Перевод в минуты

df['duration']=df['duration']/60
df['default']=df['default'].map({'no':0,'yes':1})

df['housing']=df['housing'].map({'no':0,'yes':1})

df['loan']=df['loan'].map({'no':0,'yes':1})

df.info()
# dummy df

ddf = pd.get_dummies(df, columns=['job', 'education', 'marital', 'contact', 'poutcome', 'month'])



ddf.info()
ddf.head().T
from sklearn.model_selection import train_test_split

X=ddf.drop('deposit',axis=1)

y=ddf['deposit']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(random_state=2019)

tree.fit(X_train, y_train)
tree.score(X_valid, y_valid)
from sklearn.metrics import accuracy_score



y_pred = tree.predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

kf = KFold(n_splits=5, shuffle=True, random_state=42)

tree = DecisionTreeClassifier()

tree_params_max_depth = {'max_depth': np.arange(2, 15)}

tree_grid = GridSearchCV(tree, tree_params_max_depth, cv=kf, scoring='accuracy')

tree_grid.fit(X_train, y_train)
tree_grid_cv_results_max_depth=tree_grid.cv_results_

tree_grid.best_estimator_
tree_grid.best_score_
best_tree = DecisionTreeClassifier(max_depth=9)

y_pred =best_tree.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)

# best_tree.score(X_valid, y_valid)
tree = DecisionTreeClassifier(max_depth=9)

tree_params_min_samples_split = {'min_samples_split': np.arange(2, 150)}

tree_grid = GridSearchCV(tree, tree_params_min_samples_split, cv=kf, scoring='accuracy')

tree_grid.fit(X_train, y_train)
tree_grid_cv_results_min_samples_split=tree_grid.cv_results_

tree_grid.best_estimator_
tree_grid.best_score_
best_tree = DecisionTreeClassifier(max_depth=9,min_samples_split=96)

y_pred =best_tree.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

tree = DecisionTreeClassifier(min_samples_split=96,max_depth=9)

tree_params_min_samples_leaf = {'min_samples_leaf': np.arange(1, 50)}

tree_grid = GridSearchCV(tree, tree_params_min_samples_leaf, cv=kf, scoring='accuracy')

tree_grid.fit(X_train, y_train)
tree_grid_cv_results_min_samples_leaf=tree_grid.cv_results_

tree_grid.best_estimator_
tree_grid.best_score_
best_tree = DecisionTreeClassifier(max_depth=9, min_samples_split=96, min_samples_leaf=5)

y_pred =best_tree.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

tree = DecisionTreeClassifier(min_samples_split=96,max_depth=9, min_samples_leaf=5)

tree_params_max_features = {'max_features': np.arange(1, X.shape[1])}

tree_grid = GridSearchCV(tree, tree_params_max_features, cv=kf, scoring='accuracy') 

tree_grid.fit(X_train, y_train)
tree_grid_cv_results_max_features=tree_grid.cv_results_

tree_grid.best_estimator_
tree_grid.best_score_
best_tree = DecisionTreeClassifier(min_samples_split=96,max_depth=9, min_samples_leaf=5, max_features=43)

y_pred =best_tree.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True,figsize=(20, 5))



ax[0].plot(tree_params_max_depth['max_depth'], tree_grid_cv_results_max_depth['mean_test_score'])

ax[0].set_xlabel('max_depth')

ax[0].set_ylabel('Mean accuracy on test set')



ax[1].plot(tree_params_min_samples_split['min_samples_split'], tree_grid_cv_results_min_samples_split['mean_test_score'])

ax[1].set_xlabel('min_samples_split')

ax[1].set_ylabel('Mean accuracy on test set')



ax[2].plot(tree_params_min_samples_leaf['min_samples_leaf'], tree_grid_cv_results_min_samples_leaf['mean_test_score'])

ax[2].set_xlabel('min_samples_leaf')

ax[2].set_ylabel('Mean accuracy on test set')



ax[3].plot(tree_params_max_features['max_features'], tree_grid_cv_results_max_features['mean_test_score'])

ax[3].set_xlabel('max_features')

ax[3].set_ylabel('Mean accuracy on test set')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

tree = DecisionTreeClassifier(min_samples_split=96,max_depth=9, min_samples_leaf=5, max_features=43)

tree_params_max_depth = {'max_depth': np.arange(2, 15)}

tree_grid = GridSearchCV(tree, tree_params_max_depth, cv=kf, scoring='accuracy')

tree_grid.fit(X_train, y_train)
tree_grid.best_estimator_
tree_grid.best_score_
pd.DataFrame(tree_grid.cv_results_).T
best_tree = DecisionTreeClassifier(min_samples_split=96,max_depth=10, min_samples_leaf=5, max_features=43)

y_pred =best_tree.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
from sklearn.tree import export_graphviz



export_graphviz(best_tree, out_file='best_tree.dot', feature_names=X.columns)

print(open('best_tree.dot').read())
import matplotlib.pyplot as plt



features = {'f'+str(i+1):name for (i, name) in zip(range(len(ddf.columns)), ddf.columns)}



# Важность признаков



importances = best_tree.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the tree

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
from sklearn. preprocessing import LabelEncoder



ddf=df.copy()

le = LabelEncoder()

le.fit(df.job)

ddf['job']=le.transform(df.job)



le = LabelEncoder()

le.fit(df.education)

ddf['education']=le.transform(df.education)



le = LabelEncoder()

le.fit(df.marital)

ddf['marital']=le.transform(df.marital)



le = LabelEncoder()

le.fit(df.contact)

ddf['contact']=le.transform(df.contact)



le = LabelEncoder()

le.fit(df.poutcome)

ddf['poutcome']=le.transform(df.poutcome)



le = LabelEncoder()

le.fit(df.month)

ddf['month']=le.transform(df.month)



ddf.info()
ddf.head()
from sklearn.model_selection import train_test_split

X=ddf.drop('deposit',axis=1)

y=ddf['deposit']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_valid)



accuracy_score(y_valid, y_pred)
# GridSearchCV

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier()

rf_params_n_estimators = {'n_estimators': np.arange(25, 450, 50)}

# rf_params_n_estimators

rf_grid = GridSearchCV(rf, rf_params_n_estimators, cv=kf, scoring='accuracy')

rf_grid.fit(X_train, y_train)
rf_grid_cv_results_n_estimators=rf_grid.cv_results_

rf_grid.best_estimator_
rf_grid.best_score_
pd.DataFrame(rf_grid.cv_results_).T
best_rf = RandomForestClassifier(n_estimators=125)

y_pred =best_rf.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)

# best_rf.score(X_valid, y_valid)
rf = RandomForestClassifier(n_estimators=125)

rf_params_max_depth = {'max_depth': np.arange(2, 15)}

rf_grid = GridSearchCV(rf, rf_params_max_depth, cv=kf, scoring='accuracy')

rf_grid.fit(X_train, y_train)
rf_grid_cv_results_max_depth=rf_grid.cv_results_

rf_grid.best_estimator_
rf_grid.best_score_
pd.DataFrame(rf_grid.cv_results_).T
best_rf = RandomForestClassifier(n_estimators=125, max_depth=14)

y_pred =best_rf.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)

# best_rf.score(X_valid, y_valid)
rf = RandomForestClassifier(n_estimators=125,max_depth=14)

rf_params_min_samples_split = {'min_samples_split': np.arange(2, 20)}

rf_grid = GridSearchCV(rf, rf_params_min_samples_split, cv=kf, scoring='accuracy')

rf_grid.fit(X_train, y_train)
rf_grid_cv_results_min_samples_split=rf_grid.cv_results_

rf_grid.best_estimator_
rf_grid.best_score_
pd.DataFrame(rf_grid.cv_results_).T
best_rf = RandomForestClassifier(n_estimators=125, max_depth=14,min_samples_split=12)

y_pred =best_rf.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)

# best_rf.score(X_valid, y_valid)
rf = RandomForestClassifier(n_estimators=125,max_depth=14,min_samples_split=12)

rf_params_min_samples_leaf = {'min_samples_leaf': np.arange(1, 50)}

rf_grid = GridSearchCV(rf, rf_params_min_samples_leaf, cv=kf, scoring='accuracy')

rf_grid.fit(X_train, y_train)
rf_grid_cv_results_min_samples_leaf=rf_grid.cv_results_

rf_grid.best_estimator_
rf_grid.best_score_
pd.DataFrame(rf_grid.cv_results_).T
best_rf = RandomForestClassifier(n_estimators=125, max_depth=14,min_samples_split=12,min_samples_leaf=4)

y_pred =best_rf.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)

# best_rf.score(X_valid, y_valid)
rf = RandomForestClassifier(n_estimators=125, max_depth=14,min_samples_split=12,min_samples_leaf=4)

rf_params_max_features = {'max_features': np.arange(2, X.shape[1])}

rf_grid = GridSearchCV(rf, rf_params_max_features, cv=kf, scoring='accuracy')

rf_grid.fit(X_train, y_train)
rf_grid_cv_results_max_features=rf_grid.cv_results_

rf_grid.best_estimator_
rf_grid.best_score_
pd.DataFrame(rf_grid.cv_results_).T
best_rf = RandomForestClassifier(n_estimators=125, max_depth=14,min_samples_split=5,min_samples_leaf=4,max_features=14)

y_pred =best_rf.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)

# best_rf.score(X_valid, y_valid)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, ncols=5, sharey=True,figsize=(25, 5))



ax[0].plot(rf_params_max_depth['max_depth'], rf_grid_cv_results_max_depth['mean_test_score'])

ax[0].set_xlabel('max_depth')

ax[0].set_ylabel('Mean accuracy on test set')



ax[1].plot(rf_params_min_samples_split['min_samples_split'], rf_grid_cv_results_min_samples_split['mean_test_score'])

ax[1].set_xlabel('min_samples_split')

ax[1].set_ylabel('Mean accuracy on test set')



ax[2].plot(rf_params_min_samples_leaf['min_samples_leaf'], rf_grid_cv_results_min_samples_leaf['mean_test_score'])

ax[2].set_xlabel('min_samples_leaf')

ax[2].set_ylabel('Mean accuracy on test set')



ax[3].plot(rf_params_max_features['max_features'], rf_grid_cv_results_max_features['mean_test_score'])

ax[3].set_xlabel('max_features')

ax[3].set_ylabel('Mean accuracy on test set')



ax[4].plot(rf_params_n_estimators['n_estimators'], rf_grid_cv_results_n_estimators['mean_test_score'])

ax[4].set_xlabel('n_estimators')

ax[4].set_ylabel('Mean accuracy on test set')
import matplotlib.pyplot as plt



features = {'f'+str(i+1):name for (i, name) in zip(range(len(ddf.columns)), ddf.columns)}



# Важность признаков



importances = best_rf.feature_importances_



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