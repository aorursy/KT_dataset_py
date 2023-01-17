# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import  matplotlib.pyplot as plt

from sklearn import preprocessing





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/creditcard.csv')
# show sample dateset

df.head()
# Display distribute of target Class

sns.countplot(df['Class'])

# Count Class == 0  and Class == 1

neg = 0

pos = 0

for i in df['Class']:

    if(i==1):

        neg=neg+1

    else:

        pos=pos+1

print(neg, pos)
# Show distribute of Amount and Time with left Amount Distribute and right Time Distribute

fig, ax = plt.subplots(1, 2, figsize=(18, 4))



sns.distplot(df['Amount'], ax=ax[0], color='r')



sns.distplot(df['Time'], ax=ax[1], color='b')


robust_scaler = preprocessing.RobustScaler()

df['Scaler_amount'] = robust_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

df['Scaler_time'] = robust_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
# Show distribute of scale between Scaler_amount and Scaler_time use Robust Scaler and Drop Amount and Time

fig, ax = plt.subplots(1, 2, figsize=(18, 4))



sns.distplot(df['Scaler_amount'], ax=ax[0], color='r')

sns.distplot(df['Scaler_time'], ax=ax[0], color='b')

df = df.drop(columns=['Amount', 'Time'], axis=1)
fraud_df = df.loc[df['Class']==1]

nonfraud_df = df.loc[df['Class']==0][:492]

normal_distributed_df = pd.concat([fraud_df, nonfraud_df], axis=0)
df = normal_distributed_df
sns.countplot(df['Class'])
fig, ax = plt.subplots(1, 1, figsize=(24, 20))

corr = df.corr()



sns.heatmap(corr, ax=ax[0])
X = df.drop('Class', axis=1)

y = df['Class']



import time

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

t0=time.time()

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)

t1=time.time()

print('TSNE took {:.2} s'.format(t1-t0))



t0 = time.time()

X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)

t1 = time.time()

print('PCA took {:.2} s'.format(t1-t0))



from sklearn.decomposition import TruncatedSVD



t0 = time.time()

X_reduced_svd = TruncatedSVD(n_components=2, random_state=42, algorithm = 'randomized').fit_transform(X.values)

t1= time.time()



print('Truncated SVD took {:.2} s'.format(t1 - t0))
fig, ax = plt.subplots(3, 1, figsize=(24, 20))

sns.scatterplot(x=X_reduced_tsne[:, 0], y=X_reduced_tsne[:, 1], ax=ax[0], hue=y)

sns.scatterplot(x=X_reduced_pca[:, 0], y=X_reduced_pca[:, 1], ax=ax[1], hue=y)

sns.scatterplot(x=X_reduced_svd[:, 0], y=X_reduced_svd[:, 1], ax=ax[2], hue=y)
X = df.drop('Class', axis=1)

y = df['Class']

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

print('Accuracy of Train {:.2} and Test: {:.2}'.format(accuracy_score(y_train, lr.fit(X_train, y_train).predict(X_train)), accuracy_score(y_test, lr.predict(X_test))))
from sklearn.svm import SVC

clf = SVC()

print('Accuracy of Train {:.2} and Test: {:.2}'.format(accuracy_score(y_train, clf.fit(X_train, y_train).predict(X_train)), accuracy_score(y_test, clf.predict(X_test))))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

print('Accuracy of Train {:.2} and Test: {:.2}'.format(accuracy_score(y_train, clf.fit(X_train, y_train).predict(X_train)), accuracy_score(y_test, clf.predict(X_test))))

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



title = 'Logistic Regression '

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

estimator = LogisticRegression()

plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv)

plt.show()
title = 'Support Vector machine '

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

estimator = SVC()

plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv)

plt.show()
title = 'Random Forest Classifier'

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

estimator = RandomForestClassifier()

plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv)

plt.show()