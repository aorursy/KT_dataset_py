# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load the Haberman Breast Cancer Survival dataset

haberman = pd.read_csv("../input/haberman.csv/haberman.csv")
# peek of the dataset

haberman.head()
# shape of the dataset

haberman.shape
# summarize each column

haberman.describe()
# check data types

haberman.dtypes
# create histograms of each variable

from matplotlib import pyplot

haberman.hist()

pyplot.show()
# check how imbalanced the dataset actually is

from collections import Counter

# summarize the class distribution

target = haberman['status'].values

counter = Counter(target)

for k,v in counter.items():

    per = v / len(target) * 100

    print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))
# retrieve numpy array

haberman = haberman.values

# split into input and output elements

X, y = haberman[:, :-1], haberman[:, -1]



# label encode the target variable to have the classes 0 and 1

from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(y)
from sklearn.metrics import brier_score_loss

from numpy import mean

from numpy import std

# calculate brier skill score (BSS)

def brier_skill_score(y_true, y_prob):

    # calculate reference brier score

    ref_probs = [0.26471 for _ in range(len(y_true))]

    bs_ref = brier_score_loss(y_true, ref_probs)

    # calculate model brier score

    bs_model = brier_score_loss(y_true, y_prob)

    # calculate skill score

    return 1.0 - (bs_model / bs_ref)
from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

# evaluate a model

def evaluate_model(X, y, model):

    # define evaluation procedure

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # define the model evaluation metric

    metric = make_scorer(brier_skill_score, needs_proba=True)

    # evaluate model

    scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)

    return scores
# summarize the loaded dataset

print(X.shape, y.shape, Counter(y))
from sklearn.dummy import DummyClassifier

# define the reference model

model = DummyClassifier(strategy='prior')

# evaluate the model

scores = evaluate_model(X, y, model)

print('Mean BSS: %.3f (%.3f)' % (mean(scores), std(scores)))
from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.gaussian_process import GaussianProcessClassifier
# define models to test

def get_models():

    models, names = list(), list()

    # LR

    models.append(LogisticRegression(solver='lbfgs'))

    names.append('LR')

    # LDA

    models.append(LinearDiscriminantAnalysis())

    names.append('LDA')

    # QDA

    models.append(QuadraticDiscriminantAnalysis())

    names.append('QDA')

    # GNB

    models.append(GaussianNB())

    names.append('GNB')

    # MNB

    models.append(MultinomialNB())

    names.append('MNB')

    # GPC

    models.append(GaussianProcessClassifier())

    names.append('GPC')

    return models, names
# define models

models, names = get_models()

results = list()

# evaluate each model

for i in range(len(models)):

    # evaluate the model and store results

    scores = evaluate_model(X, y, models[i])

    results.append(scores)

    # summarize and store

    print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))

    # plot the results

pyplot.boxplot(results, labels=names, showmeans=True)

pyplot.show()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
# define models to test

def get_models():

    models, names = list(), list()

    # LR

    models.append(LogisticRegression(solver='lbfgs'))

    names.append('LR')

    # LDA

    models.append(LinearDiscriminantAnalysis())

    names.append('LDA')

    # QDA

    models.append(QuadraticDiscriminantAnalysis())

    names.append('QDA')

    # GNB

    models.append(GaussianNB())

    names.append('GNB')

    # GPC

    models.append(GaussianProcessClassifier())

    names.append('GPC')

    return models, names
# define models

models, names = get_models()

results = list()

# evaluate each model

for i in range(len(models)):

    # create a pipeline

    pipeline = Pipeline(steps=[('t', StandardScaler()),('m',models[i])])

    # evaluate the model and store results

    scores = evaluate_model(X, y, pipeline)

    results.append(scores)

    # summarize and store

    print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))

# plot the results

pyplot.boxplot(results, labels=names, showmeans=True)

pyplot.show()
from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import MinMaxScaler
# define models to test

def get_models():

    models, names = list(), list()

    # LR

    models.append(LogisticRegression(solver='lbfgs'))

    names.append('LR')

    # LDA

    models.append(LinearDiscriminantAnalysis())

    names.append('LDA')

    # GPC

    models.append(GaussianProcessClassifier())

    names.append('GPC')

    return models, names
# define models

models, names = get_models()

results = list()

# evaluate each model

for i in range(len(models)):

    # create a pipeline

    steps = [('t1', MinMaxScaler()), ('t2', PowerTransformer()),('m',models[i])]

    pipeline = Pipeline(steps=steps)

    # evaluate the model and store results

    scores = evaluate_model(X, y, pipeline)

    results.append(scores)

    # summarize and store

    print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))

# plot the results

pyplot.boxplot(results, labels=names, showmeans=True)

pyplot.show()
# fit the model

steps = [('t1', MinMaxScaler()),('t2', PowerTransformer()),('m',LogisticRegression(solver='lbfgs'))]

model = Pipeline(steps=steps)

model.fit(X, y)

# some survival cases

print('Survival Cases:')

data = [[31,59,2], [31,65,4], [34,60,1]]

for row in data:

    # make prediction

    yhat = model.predict_proba([row])

    # get percentage of survival

    p_survive = yhat[0, 0] * 100

    # summarize

    print('>data=%s, Survival=%.3f%%' % (row, p_survive))

# some non-survival cases

print('Non-Survival Cases:')

data = [[44,64,6], [34,66,9], [38,69,21]]

for row in data:

    # make prediction

    yhat = model.predict_proba([row])

    # get percentage of survival

    p_survive = yhat[0, 0] * 100

    # summarize

    print('>data=%s, Survival=%.3f%%' % (row, p_survive))