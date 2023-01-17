import pandas as pd

import numpy as np



data = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv").values

#The first column is the class of an example

trainY = data[:, 0]

#Removing first row(with classes) from x

trainX = np.delete(data, 0, axis=1)
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import linear_model

import warnings

warnings.filterwarnings('ignore') 



#log stands for sigmoid error function(Logistic regression)

sgd = linear_model.SGDClassifier(max_iter=100, tol=1e-3, loss='log', n_jobs=10)

lbfgs = linear_model.LogisticRegression(max_iter=100, solver='lbfgs',multi_class='multinomial', n_jobs=10);



#Splitting dataset for training and testing

X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.2, random_state=42)





sgd.fit(X_train, y_train)

lbfgs.fit(X_train, y_train)



print("Test score for SGD Log Regression is " + str(sgd.score(X_test, y_test)))

print("Test score for lbfgs Log Regression is " + str(lbfgs.score(X_test, y_test)))
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



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

    plt.figure()

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

%matplotlib notebook

#Splitting data to 5 chunks for cross-validation

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

#Choosing the linear model

est = linear_model.LogisticRegression(max_iter=100, random_state=0, solver='lbfgs',multi_class='multinomial', n_jobs=4);

plot_learning_curve(est, "Learning curve for Logistic Regression, C=1.0", trainX, trainY, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()
%matplotlib notebook

#We will be varying Regularization constant from 10^-8 to 10

parameters = {'C':np.logspace(-8, 1, 10)}

sgd = linear_model.LogisticRegression(max_iter=100, random_state=0, solver='lbfgs',multi_class='multinomial', n_jobs=5)

#Set up the GridSearchCV with 5fold configuration

clf = GridSearchCV(sgd, parameters, cv=5, n_jobs=3)

clf.fit(trainX, trainY)



scores = clf.cv_results_['mean_test_score']



plt.xlabel("C")

plt.ylabel("Score")

plt.xscale('log')

plt.title("Validation curve for sklearn lbfgs Logistic regression classifier")

plt.plot(np.logspace(-8, 1, 10), scores, label="E for SGD logistic regression")

plt.legend(loc="best")

plt.show()



print("best score: " + str(clf.best_score_))
from sklearn import metrics



y_pred = lbfgs.predict(X_train)

print(metrics.classification_report(y_train, y_pred))

print(metrics.confusion_matrix(y_train, y_pred))