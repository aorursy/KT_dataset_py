import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

simplefilter(action='ignore', category=DeprecationWarning)



# load data

data = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')

data.head()
# concise way to display stats in the data including nulls and skewness

def data_stats(df):

    lines = df.shape[0]

    d_types = df.dtypes

    counts = df.apply(lambda x: x.count())

    unique = df.apply(lambda x: x.unique().shape[0])

    nulls = df.isnull().sum()

    missing_ratio = (df.isnull().sum()/lines)*100

    skewness = df.skew()

    col_names = ['dtypes', 'counts', 'unique', 'nulls', 'missing_ratio', 'skewness']

    temp = pd.concat([d_types, counts, unique, nulls, missing_ratio, skewness], axis=1)

    temp.columns = col_names

    return temp





stats = data_stats(data)

stats
col_names = ['mean_IP', 'std_IP', 'kurt_IP', 'skew_IP', 'mean_DMSNR', 'std_DMSNR', 'kurt_DMSNR', 'skew_DMSNR', 'target_class']

data.columns = col_names
data.describe()
fig = plt.figure(figsize=(8,6))

sns.countplot(data['target_class'])
# pairplot

sns.pairplot(data, 

             vars=['mean_IP', 'std_IP', 'kurt_IP', 'skew_IP', 'mean_DMSNR', 'std_DMSNR', 'kurt_DMSNR', 'skew_DMSNR'],

             hue='target_class')
# heatmap

fig = plt.figure(figsize=(12, 10))

sns.heatmap(data.corr(), annot=True)
# plot distributions

def plot_dists(df, col_1, col_2='target_class'):

    fig, axis = plt.subplots(1, 2, figsize=(16, 5))

    

    sns.distplot(df[col_1], ax=axis[0])

    axis[0].set_title('distribution of {}. Skewness = {:.4f}'.format(col_1 ,df[col_1].skew()))

    

    sns.violinplot(x=col_2, y=col_1, data=data, ax=axis[1], inner='quartile')

    axis[1].set_title('violin of {}, split by target'.format(col_1))

    plt.show()

    

for col in data.columns[:-1]:

    plot_dists(data, col)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.base import clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split
# score function

def cv_score(model, features, label, folds):

    cv=KFold(n_splits=folds, shuffle=True)

    cv_estimate = cross_val_score(model, features, label, cv=cv, scoring='roc_auc', n_jobs=4)

    mean = np.mean(cv_estimate)

    std = np.std(cv_estimate)

    return mean, std
# set up dataframe for estimator evaluation

indx_estimators = ['LogisticRegression', 'SVC', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'RandomForestClassifier', 'MPLClassifier']

column_names = ['roc_auc_mean', 'roc_auc_std']

results = pd.DataFrame(columns=column_names, index=indx_estimators)
# initialize models with balanced class_weight as the target is unbalanced

log_mod = LogisticRegression(class_weight='balanced')

svc_mod = SVC(probability=True, class_weight='balanced')

ada_mod = AdaBoostClassifier()

gbc_mod = GradientBoostingClassifier()

rfc_mod = RandomForestClassifier(class_weight='balanced')

mlp_mod = MLPClassifier()
# We need to transform and scale the skewed data. because there is over 1000 samples we can use the quantile transformer.

from sklearn.preprocessing import QuantileTransformer

X = data[['mean_IP', 'std_IP', 'kurt_IP', 'skew_IP', 'mean_DMSNR', 'std_DMSNR', 'kurt_DMSNR', 'skew_DMSNR']].values

y = data['target_class'].values



X = QuantileTransformer(output_distribution='normal').fit_transform(X)



print('Features shape: {}'.format(X.shape))

print('Label shape   : {}'.format(y.shape))
models = [log_mod, svc_mod, ada_mod, gbc_mod, rfc_mod, mlp_mod]

for name, mod in zip(indx_estimators, models):

    mean, std = cv_score(mod, X, y, 10)

    results.loc[name,'roc_auc_mean'] = mean

    results.loc[name, 'roc_auc_std'] = std

    

results
rf = RandomForestClassifier()

rf.fit(data[data.columns[:-1]].values, data['target_class'].values)



feature_import = pd.Series(rf.feature_importances_, index=data.columns[:-1]).sort_values(ascending=False)

feature_import



fig = plt.figure(figsize=(18, 8))

sns.barplot(x=feature_import, y=feature_import.index)

plt.title('Feature importances')

plt.xlabel('Score')

plt.show()
from sklearn.model_selection import GridSearchCV, learning_curve

from sklearn import metrics
# model comparison dataframe on test data

index_cols = ['LogisticRegression', 'SVC', 'AdaBoostClassifier', 'GradientBoosting', 'RandomForestClassifier', 'MPLClassifier']

colnames =['roc_auc', 'accuracy', 'precision', 'recall']

results_final = pd.DataFrame(columns=colnames, index=index_cols)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
log_mod = LogisticRegression(class_weight='balanced')

log_mod.get_params()
cv=KFold(n_splits=10, shuffle=True)

param_grid={'C':[0.1, 1, 10, 100, 1000]}



grid_lin_mod = GridSearchCV(estimator=log_mod,

                           param_grid=param_grid,

                           scoring='roc_auc',

                           cv=cv, 

                           return_train_score=True,

                           n_jobs=4,

                           verbose=1)



grid_lin_mod.fit(X, y)

lin_mod_best = grid_lin_mod.best_estimator_

print('GridSearchCV Best Score: {:.4f}'.format(grid_lin_mod.best_score_))

print('\nTuned HyperParameter       value')

print('C                           {}'.format(grid_lin_mod.best_estimator_.C))
def score_model(probs, threshold):

    return np.array([1 if x >= threshold else 0 for x in probs[:,1]])



def print_metrics(labels, probs, threshold):

    scores = score_model(probs, threshold)

    mets = metrics.precision_recall_fscore_support(labels, scores)

    conf = metrics.confusion_matrix(labels, scores)

    print('                 Confusion matrix')

    print('                 Score positive    Score negative')

    print('Actual positive    {:6d}             {:6d}'.format(conf[0,0], conf[0,1]))

    print('Actual negative    {:6d}             {:6d}'.format(conf[1,0], conf[1,1]))

    print('')

    print('Accuracy        {:.4f}'.format(metrics.accuracy_score(labels, scores)))

    print('AUC             {:.4f}'.format(metrics.roc_auc_score(labels, probs[:,1])))

    print('Macro precision {:.4f}'.format(float((float(mets[0][0]) + float(mets[0][1]))/2.0)))

    print('Macro recall    {:.4f}'.format(float((float(mets[1][0]) + float(mets[1][1]))/2.0)))

    print(' ')

    print('           Positive      Negative')

    print('Num case   {:6d}         {:6d}'.format(mets[3][0], mets[3][1]))

    print('Precision  {:.4f}         {:.4f}'.format(mets[0][0], mets[0][1]))

    print('Recall     {:.4f}         {:.4f}'.format(mets[1][0], mets[1][1]))

    print('F1         {:.4f}         {:.4f}'.format(mets[2][0], mets[2][1]))



def plot_auc(labels, probs):

    fpr, tpr, threshold = metrics.roc_curve(labels, probs[:,1])

    auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, color = 'blue', label = 'AUC = {:.4f}'.format(auc))

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

lin_mod_best.fit(X_train, y_train)

probabilities = lin_mod_best.predict_proba(X_test)



# add to results_final

score = score_model(probabilities, .5)

results_final.loc['LogisticRegression',:] = [metrics.roc_auc_score(y_test, probabilities[:,1]), 

                                            metrics.accuracy_score(y_test, score), 

                                            metrics.precision_score(y_test, score),

                                            metrics.recall_score(y_test, score)]



print_metrics(y_test, probabilities, .5)      

plot_auc(y_test, probabilities)
svc_mod = SVC(probability=True, class_weight='balanced')

svc_mod.get_params()
cv=KFold(n_splits=3, shuffle=True)

param_grid = [{'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},

            {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1.0/50.0, 1.0/200.0, 1.0/500.0, 1.0/1000.0], 'kernel': ['rbf']}]



svc_class_grid = GridSearchCV(estimator=svc_mod,

                           param_grid=param_grid,

                           cv=cv,

                           scoring='roc_auc',

                           return_train_score=True,

                           n_jobs=4,

                           verbose=1)



svc_class_grid.fit(X, y)

svc_best_params = svc_class_grid.best_estimator_

print('GridsearchCV Best Score: ' + str(svc_class_grid.best_score_))

print('\nTested HyperParameters           Values')

print('kernal:                              {}'.format(svc_class_grid.best_estimator_.kernel))

print('gamma:                               {}'.format(svc_class_grid.best_estimator_.gamma))

print('C:                                   {}'.format(svc_class_grid.best_estimator_.C))
svc_best_params.fit(X_train, y_train)

probabilities = svc_best_params.predict_proba(X_test)



# add to results_final

score = score_model(probabilities, .5)

results_final.loc['SVC',:] = [metrics.roc_auc_score(y_test, probabilities[:,1]), 

                              metrics.accuracy_score(y_test, score), 

                              metrics.precision_score(y_test, score),

                              metrics.recall_score(y_test, score)]



print_metrics(y_test, probabilities, .5)      

plot_auc(y_test, probabilities)
ad_clf = AdaBoostClassifier(DecisionTreeClassifier(class_weight='balanced'))

ad_clf.get_params()
cv=KFold(n_splits=3, shuffle=True)

param_grid = {'base_estimator__max_depth' :[1, 2, 5],

              'base_estimator__min_samples_split' :[2, 3 ,5],

              'base_estimator__min_samples_leaf' :[2, 3, 5 ,10],

              'n_estimators' :[10, 20, 50, 100],

              'learning_rate' :[0.001, 0.01, 0.1, 1]}







ad_clf_grid = GridSearchCV(estimator=ad_clf,

                        param_grid=param_grid,

                        cv=cv,

                        scoring='roc_auc',

                        return_train_score=True,

                        n_jobs=4,

                        verbose=1)



ad_clf_grid.fit(X, y)

ada_best_params = ad_clf_grid.best_estimator_

print('GridsearchCV Best Score: {}'.format(ad_clf_grid.best_score_))

print('\nTested HyperParameters        Values')

print('base_estimator__max_depth:         {}'.format(ad_clf_grid.best_estimator_.base_estimator.max_depth))

print('base_estimator__min_samples_split: {}'.format(ad_clf_grid.best_estimator_.base_estimator.min_samples_split))

print('base_estimator__min_samples_leaf:  {}'.format(ad_clf_grid.best_estimator_.base_estimator.min_samples_leaf))

print('n_estimators:                      {}'.format(ad_clf_grid.best_estimator_.n_estimators))

print('learning_rate:                     {}'.format(ad_clf_grid.best_estimator_.learning_rate))
ada_best_params.fit(X_train, y_train)

probabilities = ada_best_params.predict_proba(X_test)



# add to results_final

score = score_model(probabilities, .5)

results_final.loc['AdaBoostClassifier',:] = [metrics.roc_auc_score(y_test, probabilities[:,1]), 

                                   metrics.accuracy_score(y_test, score), 

                                   metrics.precision_score(y_test, score),

                                   metrics.recall_score(y_test, score)]



print_metrics(y_test, probabilities, .5)

plot_auc(y_test, probabilities)
gb_clf = GradientBoostingClassifier()

gb_clf.get_params()
cv=KFold(n_splits=3, shuffle=True)

param_grid = {'n_estimators': [10, 20, 50, 100, 500],

             'max_depth': [3, 5, 8, 15],

             'min_samples_split': [2, 5, 8, 10],

             'min_samples_leaf': [1, 2, 3, 5],

             'learning_rate': [0.001, 0.01, 0.1, 1]}



gb_clf_grid = GridSearchCV(estimator=gb_clf,

                        param_grid=param_grid,

                        cv=cv,

                        scoring='roc_auc',

                        return_train_score=True,

                        n_jobs=4,

                        verbose=1)



gb_clf_grid.fit(X, y)

gb_best_params = gb_clf_grid.best_estimator_

print('GridsearchCV Best Score: {}'.format(gb_clf_grid.best_score_))

print('\nTested HyperParameters   Values')

print('n_estimators:               {}'.format(gb_clf_grid.best_estimator_.n_estimators))

print('max_depth:                  {}'.format(gb_clf_grid.best_estimator_.max_depth))

print('min_samples_split:          {}'.format(gb_clf_grid.best_estimator_.min_samples_split))

print('min_samples_leaf:           {}'.format(gb_clf_grid.best_estimator_.min_samples_leaf))

print('learning_rate:              {}'.format(gb_clf_grid.best_estimator_.learning_rate))
gb_best_params.fit(X_train, y_train)

probabilities = gb_best_params.predict_proba(X_test)



# add to results_final

score = score_model(probabilities, .5)

results_final.loc['GradientBoosting',:] = [metrics.roc_auc_score(y_test, probabilities[:,1]), 

                                           metrics.accuracy_score(y_test, score), 

                                           metrics.precision_score(y_test, score),

                                           metrics.recall_score(y_test, score)]



print_metrics(y_test, probabilities, .5)      

plot_auc(y_test, probabilities)
rf_clf = RandomForestClassifier(class_weight='balanced')

rf_clf.get_params()
cv=KFold(n_splits=3, shuffle=True)

param_grid = {'n_estimators': [10, 20, 50, 100, 500],

             'max_depth': [3, 5, 8, 15, 50],

             'min_samples_split': [2, 5, 8, 10],

             'min_samples_leaf': [1, 2, 3, 5]}



rf_clf_grid = GridSearchCV(estimator=rf_clf,

                        param_grid=param_grid,

                        cv=cv,

                        scoring='roc_auc',

                        return_train_score=True,

                        n_jobs=4,

                        verbose=1)



rf_clf_grid.fit(X, y)

rf_best_params = rf_clf_grid.best_estimator_

print('GridsearchCV Best Score: {}'.format(rf_clf_grid.best_score_))

print('\nTested HyperParameters   Values')

print('n_estimators:               {}'.format(rf_clf_grid.best_estimator_.n_estimators))

print('max_depth:                  {}'.format(rf_clf_grid.best_estimator_.max_depth))

print('min_samples_split:          {}'.format(rf_clf_grid.best_estimator_.min_samples_split))

print('min_samples_leaf:           {}'.format(rf_clf_grid.best_estimator_.min_samples_leaf))
rf_best_params.fit(X_train, y_train)

probabilities = rf_best_params.predict_proba(X_test)



# add to results_final

score = score_model(probabilities, .5)

results_final.loc['RandomForestClassifier',:] = [metrics.roc_auc_score(y_test, probabilities[:,1]), 

                                           metrics.accuracy_score(y_test, score), 

                                           metrics.precision_score(y_test, score),

                                           metrics.recall_score(y_test, score)]



print_metrics(y_test, probabilities, .5)      

plot_auc(y_test, probabilities)
mlp_clf = MLPClassifier()

mlp_clf.get_params()
cv=KFold(n_splits=3, shuffle=True)

param_grid = {'hidden_layer_sizes' :[(50, 50, 10), (50, 100, 10), (100,)],

             'solver' :['lbfgs', 'adam'],

             'alpha' :[.000001, .00001, .0001, .001],

             'beta_1' :[.8, .9, .99],

             'beta_2' :[.8, .9, .99, .999]}



mlp_clf_grid = GridSearchCV(estimator=mlp_clf,

                           param_grid=param_grid,

                           cv=cv,

                           scoring='roc_auc',

                           return_train_score=True,

                           n_jobs=4,

                           verbose=1)



mlp_clf_grid.fit(X, y)

mlp_best_params = mlp_clf_grid.best_estimator_

print('GridSearchCV Best Score:  {}'.format(mlp_clf_grid.best_score_))

print('\nTested HyperParameters   Values')

print('hidden_layer_sizes:          {}'.format(mlp_clf_grid.best_estimator_.hidden_layer_sizes))

print('solver:                      {}'.format(mlp_clf_grid.best_estimator_.solver))

print('alpha:                       {}'.format(mlp_clf_grid.best_estimator_.alpha))

print('beta_1:                      {}'.format(mlp_clf_grid.best_estimator_.beta_1))

print('beta_2:                      {}'.format(mlp_clf_grid.best_estimator_.beta_2))
mlp_best_params.fit(X_train, y_train)

probabilities = mlp_best_params.predict_proba(X_test)



# add to results_final

score = score_model(probabilities, .5)

results_final.loc['MPLClassifier',:] = [metrics.roc_auc_score(y_test, probabilities[:,1]), 

                                           metrics.accuracy_score(y_test, score), 

                                           metrics.precision_score(y_test, score),

                                           metrics.recall_score(y_test, score)]



print_metrics(y_test, probabilities, .5)      

plot_auc(y_test, probabilities)
def plot_learning_curve(estimator, title, X, y, axes, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    



    axes.set_title(title)

    if ylim is not None:

        axes.set_ylim(*ylim)

    axes.set_xlabel("Training examples")

    axes.set_ylabel("Score")



    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    # Plot learning curve

    axes.grid(color='k')

    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,

                         train_scores_mean + train_scores_std, alpha=0.1,

                         color="r")

    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,

                         test_scores_mean + test_scores_std, alpha=0.1,

                         color="g")

    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",

                 label="Training score")

    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",

                 label="Cross-validation score")

    axes.legend(loc="best")

    return plt





cv=KFold(n_splits=10, shuffle=True)



fig, axes = plt.subplots(3, 2, figsize=(16, 20))



title = "learning Curves (LogisticRegression)"

estimator = clone(lin_mod_best)

plot_learning_curve(estimator, title, X, y, axes=axes[0, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4)



title = "Learning Curves (SVM Classifier)"

estimator = clone(svc_best_params)

plot_learning_curve(estimator, title, X, y, axes=axes[0, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4)



title = "Learning Curves (AdaBoostClassifier)"

estimator = clone(ada_best_params)

plot_learning_curve(estimator, title, X, y, axes=axes[1, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4)



title = "Learning Curves (GradientBoostingClassifier)"

estimator = clone(gb_best_params)

plot_learning_curve(estimator, title, X, y, axes=axes[1, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4)



title = "Learning Curves (RandomForestClassifier)"

estimator = clone(rf_best_params)

plot_learning_curve(estimator, title, X, y, axes=axes[2, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4)



title = "Learning Curves (MPLClassifier)"

estimator = clone(mlp_best_params)

plot_learning_curve(estimator, title, X, y, axes=axes[2, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4)



plt.show()



# reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
results_final