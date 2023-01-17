import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics

total = pd.read_csv("../input/weekly111/WeeklyActivationReport2.csv")

normaltotal = total[1:]

normaltotal 
corrmat = normaltotal.corr()

plt.subplots(figsize=(25,25))

sns.heatmap(corrmat, vmax=0.9, annot = True ,square=True)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
train = normaltotal[:12]

test = normaltotal[12:]

test["Marketing Spend"] = 3000

test["Traffic"] = 25500

test["Buy Now Events"] = 4150

test


rforest = RandomForestClassifier()

LinearModel = LinearRegression()
X_train = train.drop(["Activations"], axis = 1)

y_train = train["Activations"]

X_test = test.drop(["Activations"], axis = 1)
rforest.fit(X_train, y_train)

rforestpredict = rforest.predict(X_test)

rforestpredict
LinearModel.fit(X_train, y_train)

LinearModelpredict = LinearModel.predict(X_test)

LinearModelpredict