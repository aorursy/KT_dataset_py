import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost

from xgboost import XGBClassifier, plot_importance

from bayes_opt import BayesianOptimization

from sklearn.model_selection import cross_val_score, train_test_split, learning_curve

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
def binary_numerical(column):

    column_list = []

    for term in column:

        if term == 'Male':

            column_list.append(1)

        elif term == 'Female':

            column_list.append(0)

        elif term == 'Yes':

            column_list.append(1)

        elif term == 'No':

            column_list.append(0)

        elif term == np.nan:

            column_list.append(np.nan)

        elif term == 'No internet service':

            column_list.append(np.nan)

        elif term == 'No phone service':

            column_list.append(np.nan)

        # the following three conditions make the function do nothing if it's already been run

        elif term == 0:

            pass

        elif term == 1:

            pass

        elif term == np.nan:

            pass

        else:

            print('error on', term)

    return column_list
# numerically encoding columns with only yes/no, male/female, or no-service (which I encode as nan)

binary_cols = ['gender', 'Partner', 'Dependents', 'OnlineSecurity',

               'OnlineBackup', 'DeviceProtection', 'TechSupport',

               'StreamingTV', 'StreamingMovies', 'PaperlessBilling',

               'MultipleLines']



for col in binary_cols:

    df[col] = binary_numerical(df[col])



# creating a design matrix from the numerically encoded rows

X = df[binary_cols]

    



# one-hot-encoding multi-category columns, and appending the results to the design matrix

multi_category_cols = ['InternetService', 'Contract', 'PaymentMethod']

# I'll also append them to a dataframe with churn alone, to inspect correlations

dummy_frame = df['Churn']

for col in multi_category_cols:

    dummies = pd.get_dummies(df[col], prefix = col)

    X = pd.concat([X, dummies], axis = 'columns')

    dummy_frame = pd.concat([dummy_frame, dummies], axis = 'columns')

dummy_frame['Churn'] = binary_numerical(dummy_frame['Churn'])

# putting 'Churn' on the end, so my heatmap looks the way I want it to

dummy_frame = dummy_frame[[c for c in dummy_frame if c not in ['Churn']] + ['Churn']]

    



    

# appending the columns with ordered numerical values

# I might be able to improve performance by scaling these to a smaller range

X['tenure'] = df['tenure']

X['MonthlyCharges'] = df['MonthlyCharges']



# creating the numerical target column for churn

df['Churn'] = binary_numerical(df['Churn'])

y = df['Churn']



X.head()
sns.set(style = 'white')



correlations = df.corr()



mask = np.zeros_like(correlations, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True



cmap = sns.diverging_palette(220, 10, as_cmap=True)



plt.subplots(figsize = (20, 16))

sns.heatmap(correlations, mask = mask, cmap = cmap, center = 0, vmax = 0.6, vmin = -0.6,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True)

plt.title('Correlations with Binary and Numerical Features', fontsize = 16);
sns.set(style = 'white')



correlations = dummy_frame.corr()



mask = np.zeros_like(correlations, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True



cmap = sns.diverging_palette(220, 10, as_cmap=True)



plt.subplots(figsize = (20, 16))

sns.heatmap(correlations, mask = mask, cmap = cmap, center = 0, vmax = 0.6, vmin = -0.6,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True)

plt.title('Correlations with One-Hot-Encoded Features', fontsize = 16);
X_train, X_test, y_train, y_test = train_test_split(X, y)
pbounds = {

           'min_child_weight': (1, 10),

           'gamma': (0.5, 5),

           'subsample': (0.6, 1.0),

           'colsample_bytree': (0.6, 1.0),

           'max_depth': (0, 20),

           'n_estimators': (10, 1000),

           'learning_rate': (0.005, 0.05)

          }



def black_box_function(min_child_weight, gamma, subsample, colsample_bytree, max_depth,

                       n_estimators, learning_rate):

    

    xgb = XGBClassifier(learning_rate = learning_rate, 

                        n_estimators = int(n_estimators), 

                        objective = 'binary:logistic',

                        nthread = -1,

                        min_child_weight = min_child_weight,

                        gamma = gamma,

                        subsample = subsample,

                        colsample_bytree = colsample_bytree,

                        max_depth = int(max_depth))

    

    scores = cross_val_score(xgb, X_train, y_train, cv = 5)

    return scores.mean()





optimizer = BayesianOptimization(f = black_box_function,

                                 pbounds = pbounds,

                                 random_state = 66)
optimizer.maximize(init_points = 6, n_iter = 60)



# the bayesian optimizer outputs floats, so integer parameters have to be converted

params = optimizer.max['params']

int_parameters = ['max_depth', 'n_estimators']

for param_name in int_parameters:

    params[param_name] = int(params[param_name])
print(params)
xgb = XGBClassifier(params = params)



scores = cross_val_score(xgb, X_train, y_train, cv = 5)

print(scores)

print(scores.mean())
xgb.fit(X_train, y_train)
fig, ax = plt.subplots(figsize = (16, 8))

plot_importance(xgb, ax = ax)

plt.title('Feature Importance', fontsize = 16, y = 1.02);
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    """

    Generate a simple plot of the test and training learning curve.



    Parameters

    ----------

    estimator : object type that implements the "fit" and "predict" methods

        An object of that type which is cloned for each validation.



    title : string

        Title for the chart.



    X : array-like, shape (n_samples, n_features)

        Training vector, where n_samples is the number of samples and

        n_features is the number of features.



    y : array-like, shape (n_samples) or (n_samples, n_features), optional

        Target relative to X for classification or regression;

        None for unsupervised learning.



    ylim : tuple, shape (ymin, ymax), optional

        Defines minimum and maximum yvalues plotted.



    cv : int, cross-validation generator or an iterable, optional

        Determines the cross-validation splitting strategy.

        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,

          - integer, to specify the number of folds.

          - :term:`CV splitter`,

          - An iterable yielding (train, test) splits as arrays of indices.



        For integer/None inputs, if ``y`` is binary or multiclass,

        :class:`StratifiedKFold` used. If the estimator is not a classifier

        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.



        Refer :ref:`User Guide <cross_validation>` for the various

        cross-validators that can be used here.



    n_jobs : int or None, optional (default=None)

        Number of jobs to run in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.

        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`

        for more details.



    train_sizes : array-like, shape (n_ticks,), dtype float or int

        Relative or absolute numbers of training examples that will be used to

        generate the learning curve. If the dtype is float, it is regarded as a

        fraction of the maximum size of the training set (that is determined

        by the selected validation method), i.e. it has to be within (0, 1].

        Otherwise it is interpreted as absolute sizes of the training sets.

        Note that for classification the number of samples usually have to

        be big enough to contain at least one sample from each class.

        (default: np.linspace(0.1, 1.0, 5))

    """

    plt.figure(figsize = (16, 8))

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



plot_learning_curve(xgb, 'Learning Curve', X_train, y_train, cv = 5);
preds = xgb.predict(X_test)

# XGBoost produces confidence scores for each value, rather than definite predictions

# so if we want to use metrics like accuracy or f1-score, we have to round those values

hard_preds = preds.round()



print('Accuracy:', accuracy_score(y_test, hard_preds))

print('f1-score:', f1_score(y_test, hard_preds))

print('Precision:', precision_score(y_test, hard_preds))

print('Recall:', recall_score(y_test, hard_preds))

print('ROC AUC:', roc_auc_score(y_test, hard_preds))
conf = confusion_matrix(y_test, hard_preds)



fig, ax = plt.subplots(figsize = (10, 8))

sns.heatmap(conf, annot = True)

plt.title('Confusion Matrix', fontsize = 18, y = 1.06)

ax.set_yticklabels(['Churn', 'No Churn'], rotation = 0, fontsize = 14)

ax.set_xticklabels(['Churn', 'No Churn'], fontsize = 14);