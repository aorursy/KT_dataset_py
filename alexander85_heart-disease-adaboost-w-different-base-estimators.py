import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 500)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

simplefilter(action='ignore', category=DeprecationWarning)
data = pd.read_csv('../input/heart-disease-uci/heart.csv')

data.head()
data.describe()
def df_stats(df):

    lines = df.shape[0]

    d_types = df.dtypes

    counts = df.apply(lambda x: x.count())

    unique = df.apply(lambda x: x.unique().shape[0])

    nulls = df.isnull().sum()

    missing_ratio = (df.isnull().sum()/lines)*100

    skew = df.skew()

    kurt = df.kurt()

    col_names = ['dtypes', 'counts', 'unique', 'nulls', 'missing_ratio', 'skewness', 'kurtosis']

    temp = pd.concat([d_types, counts, unique, nulls, missing_ratio, skew, kurt], axis=1)

    temp.columns = col_names

    return temp

    

st = df_stats(data)

st
sns.pairplot(data, vars=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], hue='target')
fig = plt.figure(figsize=(18, 10))

sns.heatmap(data.corr(), annot=True)
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']



fig, ax = plt.subplots(4, 2, figsize=(15, 25))

fig.suptitle('Bar Plots of Categorical Data', fontsize=20)



def plot_cats(df, cols, target, ax):

    pos = 1

    sns.countplot(x=target, data=df, ax=ax[0, 0])

    sns.countplot(x=cols[0], hue=target, data=df, ax=ax[0, 1])

    for i in range(1, ax.shape[0]):

        for j in range(ax.shape[1]):

            sns.countplot(x=cols[pos], hue=target, data=df, ax=ax[i, j])

            pos += 1

            

    plt.show()      

        

plot_cats(data, cat_cols, 'target', ax)
def plot_numeric(df, cols, target):

    for col in cols:

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        sns.distplot(a=df[col], ax=ax[0])

        ax[0].set_title('distribution of {}, skew={:.4f}'.format(col, df[col].skew()))

        sns.boxenplot(data=df, x=target, y=col, ax=ax[1])

        ax[1].set_title('Boxen Plot Split by Target')

    plt.show()

        

plot_numeric(data, num_cols, 'target')
from sklearn.preprocessing import PowerTransformer, StandardScaler

data[['oldpeak']] = PowerTransformer(method='yeo-johnson').fit_transform(data[['oldpeak']])

data[['age', 'trestbps', 'chol', 'thalach']] = StandardScaler().fit_transform(data[['age', 'trestbps', 'chol', 'thalach']])
data.head()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(data[data.columns[:-1]].values, data['target'].values)



fig = plt.figure(figsize=(18, 8))

importance = pd.Series(rf.feature_importances_, index=data.columns[:-1]).sort_values(ascending=False)

sns.barplot(x=importance, y=importance.index)

plt.title('Feature Importance')

plt.xlabel('Score')

plt.show()
data[['cp','restecg', 'slope', 'ca', 'thal']] = data[['cp','restecg', 'slope', 'ca', 'thal']].astype(str)

data = pd.get_dummies(data)

data.head()
y = data['target'].values

X = data.drop(['target'], axis=1).values

print('Label shape:    {}'.format(y.shape))

print('Features shape: {}'.format(X.shape))
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression



ada_base = AdaBoostClassifier()

ada_deci = AdaBoostClassifier(DecisionTreeClassifier())

ada_extr = AdaBoostClassifier(ExtraTreeClassifier())

ada_logr = AdaBoostClassifier(LogisticRegression())

ada_svml = AdaBoostClassifier(SVC(probability=True , kernel='linear'))
from sklearn.model_selection import StratifiedKFold, cross_val_score

def cv_score_model(mod, X, y, folds, scoring):

    cv = StratifiedKFold(n_splits=folds, shuffle=True)

    cv_estimate = cross_val_score(mod, X, y, cv=cv, scoring=scoring, n_jobs=4)

    return np.mean(cv_estimate), np.std(cv_estimate)
models = [ada_base, ada_deci, ada_extr, ada_logr, ada_svml]

model_names = ['Base', 'DecisonTree', 'ExtraTree', 'LogisticRegression', 'SVC']



def fill_results_df(mod_list, name_list, scoring_list, X, y, folds):

    

    results = pd.DataFrame(index=name_list)

    for score in scoring_list:

        sc_mean = '{}_mean'.format(score)

        sc_std = '{}_std'.format(score)

        for name, model in zip(name_list, mod_list):

            mean, std = cv_score_model(model, X, y, folds, score)

            results.loc[name, sc_mean] = mean

            results.loc[name, sc_std] = std

    

    return results

s = ['roc_auc', 'accuracy', 'precision', 'recall']

r = fill_results_df(models, model_names, s, X, y, 10)

print('Results from untuned classifiers')

r
from sklearn.base import clone

from sklearn.model_selection import GridSearchCV
base = clone(ada_base)

base.get_params()
cv=StratifiedKFold(n_splits=10, shuffle=True)



param_grid={'n_estimators' :[50, 100, 250, 500, 750, 1000],

            'learning_rate' :[0.0001, 0.001, 0.01, 0.1, 1]}





base_grid = GridSearchCV(estimator=base,

                        param_grid=param_grid,

                        cv=cv,

                        scoring='roc_auc',

                        return_train_score=True,

                        n_jobs=4,

                        verbose=1)



base_grid.fit(X, y)

base_best_mod = base_grid.best_estimator_
print('For Adaboost with default decision stump as base estimator\n')

print('Best GridSearchCV Score roc_auc {}'.format(base_grid.best_score_))

print('Hyperparameters                 Values')

print('n_estimators:                    {}'.format(base_grid.best_estimator_.n_estimators))

print('learning_rate:                   {}'.format(base_grid.best_estimator_.learning_rate))
deci = clone(ada_deci)

deci.get_params()
cv=StratifiedKFold(n_splits=10, shuffle=True)



param_grid = {'base_estimator__max_depth' :[1, 2, 5],

              'base_estimator__min_samples_split' :[2, 3 ,5],

              'base_estimator__min_samples_leaf' :[2, 3, 5 ,10],

              'n_estimators' :[10, 50, 100, 250, 500, 750, 1000],

              'learning_rate' :[0.0001, 0.001, 0.01, 0.1, 1]}





deci_grid = GridSearchCV(estimator=deci,

                        param_grid=param_grid,

                        cv=cv,

                        scoring='roc_auc',

                        return_train_score=True,

                        n_jobs=4,

                        verbose=1)



deci_grid.fit(X, y)

deci_best_mod = deci_grid.best_estimator_
print('For Adaboost with decision tree as base estimator\n')

print('Best GridSearchCV roc_auc Score {}'.format(deci_grid.best_score_))

print('Hyperparameters                   Values')

print('base_estimator__max_depth:          {}'.format(deci_grid.best_estimator_.base_estimator.max_depth))

print('base_estimator__min_samples_split:  {}'.format(deci_grid.best_estimator_.base_estimator.min_samples_split))

print('base_estimator__min_samples_leaf:   {}'.format(deci_grid.best_estimator_.base_estimator.min_samples_leaf))

print('n_estimators:                       {}'.format(deci_grid.best_estimator_.n_estimators))

print('learning_rate:                      {}'.format(deci_grid.best_estimator_.learning_rate))
extr = clone(ada_extr)

extr.get_params()
cv=StratifiedKFold(n_splits=10, shuffle=True)



param_grid = {'base_estimator__criterion' :['gini', 'entropy'],

              'base_estimator__max_depth' :[1, 2, 5],

              'base_estimator__min_samples_split' :[2, 3 ,5],

              'base_estimator__min_samples_leaf' :[2, 3, 5 ,10],

              'n_estimators' :[10, 50, 100, 250, 500, 750, 1000],

              'learning_rate' :[0.0001, 0.001, 0.01, 0.1, 1]}







extr_grid = GridSearchCV(estimator=extr,

                        param_grid=param_grid,

                        cv=cv,

                        scoring='roc_auc',

                        return_train_score=True,

                        n_jobs=4,

                        verbose=1)



extr_grid.fit(X, y)

extr_best_mod = extr_grid.best_estimator_
print('For Adaboost with extra tree as base estimator\n')

print('Best GridSearchCV roc_auc Score {}'.format(extr_grid.best_score_))

print('Hyperparameters                   Values')

print('base_estimator__criterion:          {}'.format(extr_grid.best_estimator_.base_estimator.criterion))

print('base_estimator__max_depth:          {}'.format(extr_grid.best_estimator_.base_estimator.max_depth))

print('base_estimator__min_samples_split:  {}'.format(extr_grid.best_estimator_.base_estimator.min_samples_split))

print('base_estimator__min_samples_leaf:   {}'.format(extr_grid.best_estimator_.base_estimator.min_samples_leaf))

print('n_estimators:                       {}'.format(extr_grid.best_estimator_.n_estimators))

print('learning_rate:                      {}'.format(extr_grid.best_estimator_.learning_rate))
logr = clone(ada_logr)

logr.get_params()
cv=StratifiedKFold(n_splits=10, shuffle=True)



param_grid = {'base_estimator__C' :[0.01, 0.1, 1, 10, 50, 100, 500, 1000],

              'n_estimators' :[10, 50, 100, 250, 500, 750, 1000],

              'learning_rate' :[0.0001, 0.001, 0.01, 0.1, 1]}







logr_grid = GridSearchCV(estimator=logr,

                        param_grid=param_grid,

                        cv=cv,

                        scoring='roc_auc',

                        return_train_score=True,

                        n_jobs=4,

                        verbose=1)



logr_grid.fit(X, y)

logr_best_mod = logr_grid.best_estimator_
print('For Adaboost with logistic regression as base estimator\n')

print('Best GridSearchCV roc_auc Score {}'.format(logr_grid.best_score_))

print('Hyperparameters           Values')

print('base_estimator__C:          {}'.format(logr_grid.best_estimator_.base_estimator.C))

print('n_estimators:               {}'.format(logr_grid.best_estimator_.n_estimators))

print('learning_rate:              {}'.format(logr_grid.best_estimator_.learning_rate))
svml = clone(ada_svml)

svml.get_params()
cv=StratifiedKFold(n_splits=10, shuffle=True)



param_grid = {'base_estimator__C' :[0.01, 0.1, 1, 10, 50, 100, 500, 1000],

              'n_estimators' :[10, 50, 100, 250, 500, 750, 1000],

              'learning_rate' :[0.0001, 0.001, 0.01, 0.1, 1]}







svml_grid = GridSearchCV(estimator=svml,

                        param_grid=param_grid,

                        cv=cv,

                        scoring='roc_auc',

                        return_train_score=True,

                        n_jobs=4,

                        verbose=1)



svml_grid.fit(X, y)

svml_best_mod = svml_grid.best_estimator_
print('For Adaboost with linear kernal SVM as base estimator\n')

print('Best GridSearchCV roc_auc Score {}'.format(svml_grid.best_score_))

print('Hyperparameters           Values')

print('base_estimator__C:          {}'.format(svml_grid.best_estimator_.base_estimator.C))

print('n_estimators:               {}'.format(svml_grid.best_estimator_.n_estimators))

print('learning_rate:              {}'.format(svml_grid.best_estimator_.learning_rate))
s = ['roc_auc', 'accuracy', 'precision', 'recall']

mod_names = ['Base', 'DecisonTree', 'ExtraTree', 'LogisticRegression', 'SVC']

mods = [clone(base_best_mod), clone(deci_best_mod), clone(extr_best_mod), clone(logr_best_mod), clone(svml_best_mod)]



final_results = fill_results_df(mods, mod_names, s, X, y, 10)

print('Results from tuned classifiers')

final_results
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)



def grade_model(probs, thresh):

    return np.array([1 if x>=thresh else 0 for x in probs[:,1]])
best_probs = []

for mod in mods:

    temp_mod = clone(mod)

    temp_mod.fit(X_train, y_train)

    best_probs.append(temp_mod.predict_proba(X_test))

    

best_scores = []

for pb in best_probs:

    best_scores.append(grade_model(pb, .5))
from sklearn.metrics import confusion_matrix, auc, roc_curve



def plot_auc(labels, probs, ax, mod_name):

    fpr, tpr, threshold = roc_curve(labels, probs[:,1])

    roc = auc(fpr, tpr)

    sns.lineplot(x=fpr, y=tpr, ax=ax, label = 'AUC = {:.4f}'.format(roc))

    sns.lineplot(x=[0, 1], y=[0, 1], ax=ax)

    ax.legend(loc = 'lower right')

    ax.set_xlim([0, 1])

    ax.set_ylim([0, 1])

    ax.set_title('Receiver Operating Characteristic for {}'.format(mod_name))

    ax.set_ylabel('True Positive Rate')

    ax.set_xlabel('False Positive Rate')

    return plt
from pandas.plotting import table



def plot_confusion(lab, scor, mod_name, ax):

    conf = confusion_matrix(lab, scor)

    tab = pd.DataFrame(conf, columns=['Score positive', 'Score negative'], index=['Actual positive', 'Actual negative'])

    t = table(ax , tab, loc='center')

    t.scale(1, 3)

    ax.set_title('Confusion matrix for {}'.format(mod_name))

    ax.axis('off')
fig, ax = plt.subplots(5, 2, figsize=(15, 30))



plot_confusion(y_test, best_scores[0], 'Base Adaboost', ax[0, 0])

plot_auc(y_test, best_probs[0], ax[0, 1], 'Base Adaboost')



plot_confusion(y_test, best_scores[1], 'DecisionTree Adaboost', ax[1, 0])

plot_auc(y_test, best_probs[1], ax[1, 1], 'DecisionTree Adaboost')



plot_confusion(y_test, best_scores[2], 'ExtraTree Adaboost', ax[2, 0])

plot_auc(y_test, best_probs[2], ax[2, 1], 'ExtraTree Adaboost')



plot_confusion(y_test, best_scores[3], 'Logistic Adaboost', ax[3, 0])

plot_auc(y_test, best_probs[3], ax[3, 1], 'Logistic Adaboost')



plot_confusion(y_test, best_scores[4], 'SVC Adaboost', ax[4, 0])

plot_auc(y_test, best_probs[4], ax[4, 1], 'SVC Adaboost')



plt.show()