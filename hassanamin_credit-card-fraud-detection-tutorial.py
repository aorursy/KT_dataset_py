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
import pandas as pd

creditcard = pd.read_csv("../input/creditcardfraud/creditcard.csv",index_col=0)


# summarize the shape of the dataset

print(creditcard.shape)
from collections import Counter

# summarize the class distribution

target = creditcard.values[:,-1]

counter = Counter(target)

for k,v in counter.items():

    per = v / len(target) * 100

    print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))
creditcard.head()
creditcard['Amount'].describe()
from matplotlib import pyplot

# drop the target variable

df = creditcard.drop('Class', axis=1)

# create a histogram plot of each numeric variable

ax = df.hist(bins=100)

# disable axis labels to avoid the clutter

for axis in ax.flatten():

    axis.set_xticklabels([])

    axis.set_yticklabels([])

# show the plot

pyplot.show()
# load the dataset

def load_dataset(full_path):

    # load the dataset as a numpy array

    data = pd.read_csv(full_path,index_col=0,low_memory=False)

    # retrieve numpy array

    data = data.values

    # split into input and output elements

    X, y = data[:, :-1], data[:, -1]

    return X, y

 
from numpy import mean

from numpy import std

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import auc

from sklearn.metrics import make_scorer

from sklearn.dummy import DummyClassifier



# calculate precision-recall area under curve

def pr_auc(y_true, probas_pred):

    # calculate precision-recall curve

    p, r, _ = precision_recall_curve(y_true, probas_pred)

    # calculate area under curve

    return auc(r, p)
# evaluate a model

def evaluate_model(X, y, model):

    # define evaluation procedure

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # define the model evaluation the metric

    metric = make_scorer(pr_auc, needs_proba=True)

    # evaluate model

    scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)

    return scores


# define the reference model

model = DummyClassifier(strategy='constant', constant=1)
# define the location of the dataset

full_path = "../input/creditcardfraud/creditcard.csv"

# load the dataset

X, y = load_dataset(full_path)
print(X)
print(y)
# evaluate the model

scores = evaluate_model(X, y, model)

# summarize performance

print('Mean PR AUC: %.3f (%.3f)' % (mean(scores), std(scores)))
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import auc

from sklearn.metrics import make_scorer

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier



# define models to test

def get_models():

    models, names = list(), list()

    # CART

    models.append(DecisionTreeClassifier())

    names.append('CART')

    # KNN

    steps = [('s',StandardScaler()),('m',KNeighborsClassifier())]

    models.append(Pipeline(steps=steps))

    names.append('KNN')

    # Bagging

    models.append(BaggingClassifier(n_estimators=100))

    names.append('BAG')

    # RF

    models.append(RandomForestClassifier(n_estimators=100))

    names.append('RF')

    # ET

    models.append(ExtraTreesClassifier(n_estimators=100))

    names.append('ET')

    return models, names
# define models

models, names = get_models()

results = list()

# evaluate each model

for i in range(len(models)):

    # evaluate the model and store results

    scores = evaluate_model(X, y, models[i])

    results.append(scores)

    # summarize performance

    print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
pyplot.boxplot(results, labels=names, showmeans=True)

pyplot.show()