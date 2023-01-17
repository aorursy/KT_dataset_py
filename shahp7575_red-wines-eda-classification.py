import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import scipy.stats as stats



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit



from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import RFECV



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
dataset = pd.read_csv('../input/winequality-red.csv')

dataset.info()
dataset.head()
dataset.quality.value_counts()
plt.figure(figsize=(10, 5))

sns.countplot(x='quality', data=dataset)

plt.xlabel('Quality', fontsize=14)

plt.ylabel('Count', fontsize=14)
dataset.describe()
dataset.hist(bins=50, figsize=(15, 15))
dataset.quality[dataset.quality < 7] = 0

dataset.quality[dataset.quality >= 7] = 1
dataset.quality.value_counts()
plt.figure(figsize=(10, 5))

sns.countplot(x='quality', data=dataset)

plt.xlabel('Quality', fontsize=14)

plt.ylabel('Count', fontsize=14)
redwine = dataset.copy()



plt.figure(figsize=(8, 5))

sns.boxplot(x='quality', y='alcohol', data=redwine)

plt.title('Boxplot for alcohol', fontsize=16)

plt.xlabel('Quality', fontsize=14)

plt.ylabel('Alcohol', fontsize=14)
plt.figure(figsize=(16, 5))



plt.subplot(121)

sns.distplot(redwine[redwine.quality==0]['alcohol'], color='r')

plt.title('Distribution of alcohol for "not good" wine', fontsize=16)

plt.xlabel('Alcohol', fontsize=14)

plt.subplot(122)

sns.distplot(redwine[redwine.quality==1]['alcohol'], color='c')

plt.title('Distribution of alcohol for good wine', fontsize=16)

plt.xlabel('Alcohol', fontsize=14)
plt.figure(figsize=(8, 5))

sns.boxplot(x='quality', y='volatile acidity', data=redwine)

plt.title('Boxplot for volatile acidity', fontsize=16)

plt.xlabel('Quality', fontsize=14)

plt.ylabel('Volatile Acidity', fontsize=14)
plt.figure(figsize=(16, 5))



plt.subplot(121)

sns.distplot(redwine[redwine.quality==0]['volatile acidity'], color='r')

plt.title('Distribution of volatile acidity for "not good" wine', fontsize=16)

plt.xlabel('Volatile Acidity', fontsize=14)

plt.subplot(122)

sns.distplot(redwine[redwine.quality==1]['volatile acidity'], color='c')

plt.title('Distribution of volatile acidity for good wine', fontsize=16)

plt.xlabel('Volatile Acidity', fontsize=14)
data = redwine.loc[:, ['fixed acidity', 'volatile acidity', 'citric acid']]

ax = sns.PairGrid(data)

ax.map_lower(sns.kdeplot)

ax.map_upper(sns.scatterplot)

ax.map_diag(sns.kdeplot)
ax = sns.jointplot(x=redwine.loc[:, 'free sulfur dioxide'], y=redwine.loc[:, 'total sulfur dioxide'], kind='reg')

ax.annotate(stats.pearsonr)
plt.figure(figsize=(12,5))



plt.subplot(121)

sns.boxplot(x='quality', y='free sulfur dioxide', data=redwine)

plt.title('Boxplot for free sulphur dioxide', fontsize=16)

plt.xlabel('Quality', fontsize=14)

plt.ylabel('Free sulfur Dioxide', fontsize=14)

plt.subplot(122)

sns.boxplot(x='quality', y='total sulfur dioxide', data=redwine)

plt.title('Boxplot for total sulphur dioxide', fontsize=16)

plt.xlabel('Quality', fontsize=14)

plt.ylabel('Total sulfur Dioxide', fontsize=14)
plt.figure(figsize=(8,5))

sns.boxplot(x='quality', y='residual sugar', data=redwine)

plt.title('Boxplot for Residual sugar', fontsize=16)

plt.xlabel('Quality', fontsize=14)

plt.ylabel('Residual Sugar', fontsize=14)
plt.figure(figsize=(8,5))

sns.boxplot(x='quality', y='chlorides', data=redwine)

plt.title('Boxplot for Chlorides', fontsize=16)

plt.xlabel('Quality', fontsize=14)

plt.ylabel('Chlorides', fontsize=14)
plt.figure(figsize=(16, 5))



plt.subplot(121)

sns.distplot(redwine[redwine.quality==0]['pH'], color='r')

plt.title('Distribution of pH for "not good" wine', fontsize=16)

plt.xlabel('pH', fontsize=14)

plt.subplot(122)

sns.distplot(redwine[redwine.quality==1]['pH'], color='c')

plt.title('Distribution of pH for good wine', fontsize=16)

plt.xlabel('pH', fontsize=14)
for col, x in zip(('k','c','g'),('fixed acidity', 'volatile acidity', 'citric acid')):

    ax = sns.jointplot(x=x, y='pH', data=redwine, kind='reg', color=col)

    ax.annotate(stats.pearsonr)

    plt.xlabel(x, fontsize=14)

    plt.ylabel('pH', fontsize=14)
plt.figure(figsize=(16, 5))



plt.subplot(121)

sns.distplot(redwine[redwine.quality==0]['density'], color='r')

plt.title('Distribution of density for "not good" wine', fontsize=16)

plt.xlabel('Density', fontsize=14)

plt.subplot(122)

sns.distplot(redwine[redwine.quality==1]['density'], color='c')

plt.title('Distribution of density for good wine', fontsize=16)

plt.xlabel('Density', fontsize=14)
for col, x in zip(('k','c'),('alcohol', 'residual sugar')):

    ax = sns.jointplot(x=x, y='density', data=redwine, kind='reg', color=col)

    ax.annotate(stats.pearsonr)

    plt.xlabel(x, fontsize=14)

    plt.ylabel('density', fontsize=14)
# Separating independent and dependent variable

redwine_X = redwine.drop('quality', axis=1)

redwine_y = redwine['quality']

col = redwine_X.columns



# Standardization

sd = StandardScaler()

redwine_X_scaled = sd.fit_transform(redwine_X)

redwine_X_scaled = pd.DataFrame(redwine_X_scaled, columns=col)



# Swarmplot for first 5 features

data = pd.concat([redwine_y, redwine_X_scaled.iloc[:, :5]], axis=1)

data = pd.melt(data, id_vars='quality', var_name='feature', value_name='value')



plt.figure(figsize=(12,6))

sns.swarmplot(x='feature', y='value', hue='quality', data=data)

plt.title('Swarmplot for first 5 features', fontsize=16)

plt.xlabel('Feature', fontsize=14)

plt.ylabel('Value', fontsize=14)
# Swarmplot for last 6 features

data = pd.concat([redwine_y, redwine_X_scaled.iloc[:, 5:]], axis=1)

data = pd.melt(data, id_vars='quality', var_name='feature', value_name='value')



plt.figure(figsize=(12,6))

sns.swarmplot(x='feature', y='value', hue='quality', data=data)

plt.title('Swarmplot for last 6 features', fontsize=16)

plt.xlabel('Feature', fontsize=14)

plt.ylabel('Value', fontsize=14)
plt.figure(figsize=(10, 5))

sns.heatmap(redwine.corr(), annot=True, fmt='.2f')
dataset_additional = dataset.copy()



dataset_additional['total_acidity'] = dataset_additional['fixed acidity'] + dataset_additional['volatile acidity']

dataset_additional['total_acidity_citric'] = dataset_additional['total_acidity'] + dataset_additional['citric acid']

dataset_additional['bound_sulphur_dioxide'] = dataset_additional['total sulfur dioxide'] - dataset_additional['free sulfur dioxide']

dataset_additional['org_minus_fail_SG'] = (dataset_additional['alcohol'] * 7.36)/1000
plt.figure(figsize=(12, 6))

sns.heatmap(dataset_additional.corr(), annot=True, fmt='.2f')
X = dataset.drop('quality', axis=1)

y = dataset['quality']



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(X, y):

    strat_train_set = dataset.loc[train_index]

    strat_test_set = dataset.loc[test_index]
strat_train_set['quality'].value_counts()/len(strat_train_set), strat_test_set['quality'].value_counts()/len(strat_test_set)
# Check it with the actual dataset proportions

dataset['quality'].value_counts()/len(dataset)
X_train = strat_train_set.drop('quality', axis=1)

y_train = strat_train_set['quality']

X_test = strat_test_set.drop('quality', axis=1)

y_test = strat_test_set['quality']
sd = StandardScaler()



X_train_scaled = sd.fit_transform(X_train)
log_reg = LogisticRegression(random_state=42)

svm_clf = SVC(random_state=42)

tree_clf = DecisionTreeClassifier(random_state=42)

forest_clf = RandomForestClassifier(random_state=42)



X_test_scaled = sd.transform(X_test)



for clf in (log_reg, svm_clf, tree_clf, forest_clf):

    predicted = cross_val_predict(clf, X_train_scaled, y_train, cv=10)

    print(clf.__class__.__name__, ': ', accuracy_score(y_train, predicted))
param_grid = [

    {'n_estimators': [10, 25, 50, 100, 150], 'max_features': [2,3,6,9]},

    {'bootstrap': [False], 'n_estimators': [10, 25, 50, 100, 150], 'max_features': [2,3,6,9]}

]



forest_clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(forest_clf, param_grid=param_grid, cv=5)

grid_search.fit(X_train_scaled, y_train)
grid_search.best_params_
attributes = X.columns

sorted(zip(grid_search.best_estimator_.feature_importances_, attributes), reverse=True)
# Feature Selection

forest_clf_2 = RandomForestClassifier()

rfecv = RFECV(forest_clf_2, step=1, cv=5, scoring='accuracy')

rfecv.fit(X_train_scaled, y_train)
print('Optimal no. of features: ', rfecv.n_features_)

print('Best Features: ', attributes[rfecv.support_])
# No. of features vs CV scores

plt.xlabel('number of features', fontsize=14)

plt.ylabel('CV Score', fontsize=14)

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
attributes_selected = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']



X_train_selected = strat_train_set[attributes_selected]

X_test_selected = strat_test_set[attributes_selected]



X_train_sel_scaled = sd.fit_transform(X_train_selected)

X_test_sel_scaled = sd.transform(X_test_selected)



grid_search.best_estimator_.fit(X_train_sel_scaled, y_train)
y_pred = grid_search.best_estimator_.predict(X_test_sel_scaled)

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)