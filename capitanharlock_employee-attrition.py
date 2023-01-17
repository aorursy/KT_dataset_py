from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from matplotlib.gridspec import GridSpec

from time import time

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.model_selection import RandomizedSearchCV, cross_validate

import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.utils import resample

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, roc_auc_score, roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

import itertools

import os



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use(['ggplot', 'seaborn'])
df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.info()
df.describe().T
df.sample(10)
data = df['Attrition'].value_counts()



_ = sns.barplot(data.index, data.values, palette='muted')



df.loc[df['Attrition'] == 'Yes', 'Attrition'] = 1

df.loc[df['Attrition'] == 'No', 'Attrition'] = 0
def PlotDists(feature, position):

    '''

    '''

    g = sns.catplot(x=feature, y='Attrition',

                    data=df, palette='muted', kind='bar', height=6, ax=position)



    g.despine(left=True)



    g = g.set_ylabels('Attrition probability')



    # This is needed, see: https://stackoverflow.com/questions/33925494/seaborn-produces-separate-figures-in-subplots

    plt.close(g.fig)
to_plot = ['BusinessTravel', 'Department', 'EducationField', 'Gender',

           'JobRole', 'MaritalStatus', 'OverTime']



fig, ax = plt.subplots(4, 2, figsize=(20, 20), sharex=False, sharey=False)



# Flatten out the axis object

ax = ax.ravel()



for i in range(7):



    PlotDists(to_plot[i], ax[i])



plt.tight_layout()

plt.show()
# How old are the employeed by gender?

print('The average age for men is:',

      df.loc[df['Gender'] == 'Male', 'Age'].mean())



print('The average age for women is:',

      df.loc[df['Gender'] == 'Female', 'Age'].mean())



g = sns.FacetGrid(df, col='Gender')

g = g.map(sns.distplot, 'Age')
# Let us explore the likelihood of quitting by gender.

g = sns.barplot(x='Gender', y='Attrition', data=df)
# I want to explore the probability of leaving based on job satisfaction by gender.

_ = sns.violinplot(x='Attrition', y='JobSatisfaction',

                   data=df, hue='Gender', palette='muted')
# How about salary? Is there any disparity between sexes?

g = sns.barplot(x='Gender', y='MonthlyIncome', data=df,

                hue='Attrition', palette='muted')
_ = sns.barplot(x='JobSatisfaction', y='MonthlyIncome',

                data=df, hue='Attrition', palette='muted')
corr = df.corr()



fig, ax = plt.subplots(figsize=(15, 15))



_ = sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, ax=ax)
df.describe(include=['O'])
to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']



df.drop(columns=to_drop, inplace=True)
# Get numerical and categorical features

numerical = df.select_dtypes(exclude=['object'])

categorical = df.select_dtypes(['object'])
# One hot encode the remaining variables.

travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}



df['BusinessTravel'] = df['BusinessTravel'].map(travel_map)



# One hot encode categorical features

df = pd.get_dummies(df, drop_first=True)
# Split the dataset

X = df.drop('Attrition', axis=1)

y = df['Attrition']



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=.2, random_state=42, stratify=y)
# Oversampling the minority class

X_train_up, y_train_up = resample(X_train[y_train == 1],

                                  y_train[y_train == 1],

                                  replace=True,

                                  n_samples=X_train[y_train == 0].shape[0],

                                  random_state=1)



X_train_up = pd.concat([X_train[y_train == 0], X_train_up])

y_train_up = pd.concat([y_train[y_train == 0], y_train_up])
# Downsample majority class

X_train_dw, y_train_dw = resample(X_train[y_train == 0],

                                  y_train[y_train == 0],

                                  replace=True,

                                  n_samples=X_train[y_train == 1].shape[0],

                                  random_state=1)



X_train_dw = pd.concat([X_train[y_train == 1], X_train_dw])

y_train_dw = pd.concat([y_train[y_train == 1], y_train_dw])
# Check the shapes of the classes

print("Original shape:", X_train.shape, y_train.shape)

print("Upsampled shape:", X_train_up.shape, y_train_up.shape)

print("Downsampled shape:", X_train_dw.shape, y_train_dw.shape)
# Check the principal components



pca = PCA(n_components=None, svd_solver="full")



scaler = StandardScaler()

scaler = scaler.fit_transform(X_train)



pca.fit(scaler)



cum_var_exp = np.cumsum(pca.explained_variance_ratio_)



plt.figure(figsize=(12, 6))



n_features = len(cum_var_exp) + 1



plt.bar(range(1, n_features), pca.explained_variance_ratio_, align="center",

        color='magenta', label="Individual explained variance")



plt.step(range(1, n_features), cum_var_exp, where="mid",

         label="Cumulative explained variance", color='blue')



plt.xticks(range(1, n_features))

plt.legend(loc="best")



plt.xlabel("Principal component index", {"fontsize": 14})

plt.ylabel("Explained variance ratio", {"fontsize": 14})

plt.title("PCA on training data", {"fontsize": 16})
print('We need', np.where(cum_var_exp > 0.90)[

      0][0], 'features to explain 90% of the variation of the data.')

print('We need', np.where(cum_var_exp > 0.95)[

      0][0], 'features to explain 95% of the variation of the data.')

print('We need', np.where(cum_var_exp > 0.99)[

      0][0], 'features to explain 99% of the variation of the data.')
# Utility function to report best scores





def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
# Preparing the parameters grid

# Number of trees in the random forest

n_estimators = [int(n) for n in np.linspace(200, 1000, 100)]



# Number of features to consider at each split

max_features = ['auto', 'sqrt', 'log2']



# Maximum number of levels in tree

max_depth = [int(n) for n in np.linspace(10, 100, 10)]

max_depth.append(None)



# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]



# Minimum number of leaf required at each node

min_samples_leaf = [1, 2, 4]



# Method of selecting each samples for training each tree

bootstrap = [True, False]



# Construct the grid

param_grid = {

    'randomforestclassifier__n_estimators': n_estimators,

    'randomforestclassifier__max_features': max_features,

    'randomforestclassifier__max_depth': max_depth,

    'randomforestclassifier__min_samples_split': min_samples_split,

    'randomforestclassifier__min_samples_leaf': min_samples_leaf,

    'randomforestclassifier__bootstrap': bootstrap

}



datasets = {'imbalanced': (X_train, y_train),

            'up_sampled': (X_train_up, y_train_up),

            'dw_sampled': (X_train_dw, y_train_dw)}



for dataset in datasets:



    pipeline = make_pipeline(StandardScaler(),

                             RandomForestClassifier(n_estimators=200,

                                                    class_weight='balanced',

                                                    random_state=42))



    n_iter_search = 20



    gs_rf = RandomizedSearchCV(pipeline, param_distributions=param_grid,

                               scoring='f1', cv=10, n_jobs=-1, refit=True, iid=False, n_iter=n_iter_search)



    start = time()



    gs_rf.fit(datasets[dataset][0], datasets[dataset][1])



    print("RandomizedSearchCV took %.2f seconds for %d candidates"

          " parameter settings." % ((time() - start), n_iter_search))



    report(gs_rf.cv_results_)



    print("The best hyperparameters for {} data:".format(dataset))

    for hyperparam in gs_rf.best_params_.keys():

        print(hyperparam[hyperparam.find("__") + 2:],

              ": ", gs_rf.best_params_[hyperparam])



    print(

        "Best 10-folds CV f1-score: {:.2f}%.".format((gs_rf.best_score_) * 100))
def plot_roc_and_conf_matrix(clf, X, y, figure_size=(15, 5)):

    '''

    Plot both confusion matrix and ROC curce on the same figure.

    Parameters:

    -----------

    clf : sklearn.estimator

        model to use for predicting class probabilities.

    X : array_like

        data to predict class probabilities.

    y : array_like

        true label vector.

    figure_size : tuple (optional)

        size of the figure.

    Returns:

    --------

    plot : matplotlib.pyplot

        plot confusion matrix and ROC curve.

    '''

    # Compute tpr, fpr, auc and confusion matrix

    fpr, tpr, thresholds = roc_curve(y, clf.predict_proba(X)[:, 1])

    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])

    conf_mat_clf = confusion_matrix(y, clf.predict(X))



    # Define figure size and figure ratios

    plt.figure(figsize=figure_size)

    gs = GridSpec(1, 2, width_ratios=(1, 2))



    # Plot confusion matrix

    ax0 = plt.subplot(gs[0])

    ax0.matshow(conf_mat_clf, cmap='coolwarm', alpha=0.2)



    for i in range(2):

        for j in range(2):

            ax0.text(x=j, y=i, s=conf_mat_clf[i, j], ha="center", va="center")



    plt.title("Confusion matrix", y=1.1, fontdict={"fontsize": 20})

    plt.xlabel("Predicted", fontdict={"fontsize": 14})

    plt.ylabel("Actual", fontdict={"fontsize": 14})



    # Plot ROC curce

    ax1 = plt.subplot(gs[1])

    ax1.plot(fpr, tpr, label="auc = {:.3f}".format(auc))

    plt.title("ROC curve", y=1, fontdict={"fontsize": 20})

    ax1.plot([0, 1], [0, 1], "r--")

    plt.xlabel("False positive rate", fontdict={"fontsize": 16})

    plt.ylabel("True positive rate", fontdict={"fontsize": 16})

    plt.legend(loc="lower right", fontsize="medium")
# Rename variables

X_train, y_train = X_train_up.copy(), y_train_up.copy()



# Free some memory

del X_train_dw, y_train_dw, X_train_up, y_train_up



# Refit the classifier with best parameters

pipeline = make_pipeline(StandardScaler(),

                         RandomForestClassifier(n_estimators=765,

                                                min_samples_split=2,

                                                min_samples_leaf=1,

                                                max_features='log2',

                                                max_depth=30,

                                                bootstrap=False,

                                                class_weight='balanced',

                                                n_jobs=-1,

                                                random_state=42))



rf = pipeline.fit(X_train, y_train)



rf_pred = rf.predict(X_test)



print('F1 score for the RandomForrest is', f1_score(y_test, rf_pred))





# Since the classes are balanced now, we can use the ROC curve to estimate the model skill. Otherwise we should have also plot a precision-recall curve.

plot_roc_and_conf_matrix(pipeline, X_test, y_test)
# Plot feature importance according to the RandomForrest calssifier

# can also use: pipeline.named_steps['predictor'].feature_importances_

important_features = pd.Series(

    data=pipeline.steps[1][1].feature_importances_, index=X_train.columns)

important_features.sort_values(ascending=False, inplace=True)



_ = sns.barplot(x=important_features.values,

                y=important_features.index, orient='h')
# Drop features that are highly correlated with each other but that are less important.

to_drop = ['JobLevel', 'TotalWorkingYears',

           'YearsAtCompany', 'PerformanceRating']



X_train_red = X_train.drop(columns=to_drop).copy()

X_test_red = X_test.drop(columns=to_drop).copy()



y_train_red = y_train.drop(columns=to_drop).copy()

y_test_red = y_test.drop(columns=to_drop).copy()
# Let us train again our classifier and see the performance with the reduced dataset.

pipeline = make_pipeline(StandardScaler(),

                         RandomForestClassifier(n_estimators=765,

                                                min_samples_split=2,

                                                min_samples_leaf=1,

                                                max_features='log2',

                                                max_depth=30,

                                                bootstrap=False,

                                                class_weight='balanced',

                                                n_jobs=-1,

                                                random_state=42))



rf = pipeline.fit(X_train_red, y_train_red)
rf_pred = rf.predict(X_test_red)





scorer = make_scorer(f1_score)



score = cross_val_score(pipeline, X_test_red, y_test_red, scoring=scorer, cv=5)



print(score.mean())



print('F1 score for the RandomForrest is', f1_score(y_test_red, rf_pred))



plot_roc_and_conf_matrix(pipeline, X_test_red, y_test_red)
important_features = pd.Series(

    data=pipeline.steps[1][1].feature_importances_, index=X_train_red.columns)

important_features.sort_values(ascending=False, inplace=True)



_ = sns.barplot(x=important_features.values,

                y=important_features.index, orient='h')
# Let us try different algorithms



clfs = []



clfs.append(LogisticRegression(random_state=42))

clfs.append(SVC(random_state=42))

clfs.append(DecisionTreeClassifier(random_state=42))

clfs.append(RandomForestClassifier(random_state=42))

clfs.append(KNeighborsClassifier())

clfs.append(LinearDiscriminantAnalysis())

clfs.append(ExtraTreesClassifier(random_state=42))



cv_results = []



for clf in clfs:



    result = cross_val_score(clf, X_train, y_train,

                             scoring='f1', cv=10, n_jobs=-1)



    cv_results.append(result)

# Plot the CV results

cv_means = []

cv_stds = []



for result in cv_results:



    cv_means.append(result.mean())

    cv_stds.append(result.std())



algs = ['LogisticRegression', 'SVC', 'DecisionTree',

        'RandomForrest', 'KNN', 'LinearDiscriminant', 'ExtraTrees']



df_results = pd.DataFrame(

    {'cv_mean': cv_means, 'cv_std': cv_stds, 'algorithm': algs})



g = sns.barplot('cv_mean', 'algorithm', data=df_results,

                palette='muted', orient='h', xerr=cv_stds)



g.set_xlabel('F1 score')

g.set_title('CV Scores')
pip_svc = make_pipeline(StandardScaler(),

                        SVC(class_weight='balanced',

                            probability=True))



svc_params = {'svc__kernel': ['rbf', 'poly'],

              'svc__gamma': [0.001, 0.01, 0.1, 1],

              'svc__C': [1, 10, 50, 100, 200, 300, 1000]}



gs_svc = RandomizedSearchCV(pip_svc, param_distributions=svc_params,

                            scoring='f1', cv=10, n_jobs=-1, refit=True, iid=False, n_iter=n_iter_search)



gs_svc.fit(X_train, y_train)



print("The best hyperparameters:")

print("-" * 25)



for hyperparam in gs_svc.best_params_.keys():



    print(hyperparam[hyperparam.find("__") + 2:],

          ": ", gs_svc.best_params_[hyperparam])



# Print CV

print('The best 10-folds CV f1-score is: {:.2f}%'.format(

    np.mean(gs_svc.best_score_) * 100))
plot_roc_and_conf_matrix(gs_svc, X_test, y_test)
dtc = DecisionTreeClassifier(class_weight='balanced')



pip_ada = make_pipeline(StandardScaler(),

                        AdaBoostClassifier(dtc, random_state=42))



ada_params = {"adaboostclassifier__base_estimator__criterion": ["gini", "entropy"],

              "adaboostclassifier__base_estimator__splitter":   ["best", "random"],

              "adaboostclassifier__algorithm": ["SAMME", "SAMME.R"],

              "adaboostclassifier__n_estimators": [1, 10, 50, 100],

              "adaboostclassifier__learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]

              }



gs_ada_dtc = RandomizedSearchCV(pip_ada, param_distributions=ada_params,

                                cv=10, scoring='f1', refit=True, iid=False, n_jobs=-1, verbose=1, n_iter=n_iter_search)



gs_ada_dtc.fit(X_train, y_train)



print("The best hyperparameters:")

print("-" * 25)



for hyperparam in gs_ada_dtc.best_params_.keys():



    print(hyperparam[hyperparam.find("__") + 2:],

          ": ", gs_ada_dtc.best_params_[hyperparam])



# Print CV

print('The 10-folds CV f1-score is: {:.2f}%'.format(

    np.mean(gs_ada_dtc.best_score_) * 100))

plot_roc_and_conf_matrix(gs_ada_dtc, X_test, y_test)
pip_rf = make_pipeline(StandardScaler(),

                       RandomForestClassifier(class_weight='balanced',

                                              random_state=42))



n_estimators = [int(n) for n in np.linspace(200, 1000, 100)]

max_features = ['auto', 'sqrt', 'log2']

max_depth = [int(n) for n in np.linspace(10, 100, 10)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]



rf_params = {'randomforestclassifier__n_estimators': n_estimators,

             'randomforestclassifier__max_features': max_features,

             'randomforestclassifier__max_depth': max_depth,

             'randomforestclassifier__min_samples_split': min_samples_split,

             'randomforestclassifier__min_samples_leaf': min_samples_leaf,

             'randomforestclassifier__bootstrap': bootstrap

             }



gs_rf = RandomizedSearchCV(pip_rf,

                           param_distributions=rf_params,

                           scoring='f1',

                           cv=10,

                           n_jobs=-1,

                           refit=True,

                           iid=False,

                           n_iter=n_iter_search)



gs_rf.fit(X_train, y_train)



print("The best hyperparameters:")

print("-" * 25)



for hyperparam in gs_rf.best_params_.keys():



    print(hyperparam[hyperparam.find("__") + 2:],

          ": ", gs_rf.best_params_[hyperparam])



# Print CV

print('The 10-folds CV f1-score is: {:.2f}%'.format(

    np.mean(gs_rf.best_score_) * 100))
plot_roc_and_conf_matrix(gs_rf, X_test, y_test)
pip_et = make_pipeline(StandardScaler(),

                       ExtraTreesClassifier(class_weight='balanced',

                                            random_state=42))



et_params = {"extratreesclassifier__max_depth": [None],

             "extratreesclassifier__max_features": [1, 3, 10],

             "extratreesclassifier__min_samples_split": [2, 3, 10],

             "extratreesclassifier__min_samples_leaf": [1, 3, 10],

             "extratreesclassifier__bootstrap": [False],

             "extratreesclassifier__n_estimators": [100, 300],

             "extratreesclassifier__criterion": ["gini"]}



gs_et = RandomizedSearchCV(pip_et,

                           param_distributions=et_params,

                           scoring='f1',

                           cv=10,

                           n_jobs=-1,

                           refit=True,

                           iid=False,

                           n_iter=n_iter_search)



gs_et.fit(X_train, y_train)



print("The best hyperparameters:")

print("-" * 25)



for hyperparam in gs_et.best_params_.keys():



    print(hyperparam[hyperparam.find("__") + 2:],

          ": ", gs_et.best_params_[hyperparam])



# Print CV

print('The 10-folds CV f1-score is: {:.2f}%'.format(

    np.mean(gs_et.best_score_) * 100))
plot_roc_and_conf_matrix(gs_et, X_test, y_test)
# important_features = pd.Series(

#     data=pipeline.steps[1][1].feature_importances_, index=X_train_red.columns)

# important_features.sort_values(ascending=False, inplace=True)



# _ = sns.barplot(x=important_features.values,

#                 y=important_features.index, orient='h')



svc_best = gs_svc.best_estimator_

dtc_best = gs_ada_dtc.best_estimator_

rf_best = gs_rf.best_estimator_

et_best = gs_et.best_estimator_



clfs = [(dtc_best, 'BDT'), (rf_best, 'RandomForrest'), (et_best, 'ExtraTree')]



fig, ax = plt.subplots(2, 2, figsize=(15, 15), sharex=True)



ax = ax.ravel()



for i in range(len(clfs)):



    important_features = pd.Series(

        data=clfs[i][0].steps[1][1].feature_importances_, index=X_train.columns)



    important_features.sort_values(ascending=False, inplace=True)



    g = sns.barplot(x=important_features.values,

                    y=important_features.index, orient='h',

                    ax=ax[i])



    g.set_xlabel('Relative Importance')

    g.set_ylabel('Features')

    g.set_title(clfs[i][1])
def plot_roc(estimators, X, y, figure_size=(16, 6)):

    """

    Plot both confusion matrix and ROC curce on the same figure.

    Parameters:

    -----------

    estimators : dict

        key, value for model name and sklearn.estimator to use for predicting

        class probabilities.

    X : array_like

        data to predict class probabilities.

    y : array_like

        true label vector.

    figure_size : tuple (optional)

        size of the figure.

    Returns:

    --------

    plot : matplotlib.pyplot

        plot confusion matrix and ROC curve.

    """

    plt.figure(figsize=figure_size)

    for estimator in estimators.keys():

        # Compute tpr, fpr, auc and confusion matrix

        fpr, tpr, thresholds = roc_curve(

            y, estimators[estimator].predict_proba(X)[:, 1])

        auc = roc_auc_score(y, estimators[estimator].predict_proba(X)[:, 1])



        # Plot ROC curce

        plt.plot(fpr, tpr, label="{}: auc = {:.3f}".format(estimator, auc))

        plt.title("ROC curve", y=1, fontdict={"fontsize": 20})

        plt.legend(loc="lower right", fontsize="medium")



    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False positive rate", fontdict={"fontsize": 16})

    plt.ylabel("True positive rate", fontdict={"fontsize": 16})
# Plot ROC curves for all classifiers

estimators = {"RF": rf_best,

              "SVC": svc_best,

              "BDT": dtc_best,

              "ETC": et_best}



plot_roc(estimators, X_test, y_test, (12, 8))



# Print out accuracy score on test data

print("The accuracy rate and f1-score on test data are:")

for estimator in estimators.keys():

    print("{}: {:.2f}%, {:.2f}%.".format(estimator,

                                         accuracy_score(

                                             y_test, estimators[estimator].predict(X_test)) * 100,

                                         f1_score(y_test, estimators[estimator].predict(X_test)) * 100))